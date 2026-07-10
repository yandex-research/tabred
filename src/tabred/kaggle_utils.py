import hashlib
import random
import shutil
import tarfile
import tempfile
from collections.abc import Iterable
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Event

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import (
    ApiDownloadDataFilesRequest,
    ApiListCompetitionsRequest,
)
from kagglesdk.competitions.types.competition_enums import CompetitionListTab
from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest
from requests import exceptions as requests_exceptions
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TransferSpeedColumn,
)
from rich.theme import Theme
from urllib3 import exceptions as urllib3_exceptions

from tabred.registry import KaggleRef, competition_slugs, dataset_names, get_preprocessed_files

_CHUNK_SIZE = 1024 * 1024
_DOWNLOAD_RETRIES = 5
_DOWNLOAD_TIMEOUT = (5, 5)


class DownloadFailed(Exception):
    pass


_DOWNLOAD_ERRORS = (
    requests_exceptions.RequestException,
    urllib3_exceptions.HTTPError,
)

console = Console(
    highlight=False,
    theme=Theme(
        {
            'bar.complete': 'dim green',
            'bar.pulse': 'dim green',
            'bar.back': 'white',
        }
    ),
)


def _kaggle_api() -> KaggleApi:
    api = KaggleApi()
    api.authenticate()
    return api


def download_preprocessed_data(
    names: Iterable[str],
    dst: Path,
    *,
    force: bool = False,
    max_workers: int = 4,
) -> dict[str, Path]:
    require_competition_access(names)

    names = dataset_names(names)
    dst.mkdir(parents=True, exist_ok=True)

    todo = []
    for name in names:
        dataset_dir = dst / name
        if dataset_dir.exists():
            if force:
                shutil.rmtree(dataset_dir)
            else:
                console.print(f'[bold]{name}[/] [yellow]already exists[/]')
                continue
        todo.append(name)

    if not todo:
        return {name: dst / name for name in names}

    with tempfile.TemporaryDirectory(dir=dst) as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        archives = download_kaggle_data(
            get_preprocessed_files(todo),
            tmp_dir,
            force=True,
            max_workers=max_workers,
        )
        for name, archive in archives.items():
            with tarfile.open(archive, 'r:gz') as tar:
                tar.extractall(dst, filter='data')
            if not (dst / name).is_dir():
                raise DownloadFailed(f'{archive.name} did not extract to expected directory {name}')

    console.print(f'[bold green]✓[/] unpacked requested data into `{dst.as_posix()}`')
    return {name: dst / name for name in names}


# This could have been a for loop with kaggle cli download,
# but I wanted the download to be faster and have a nice ui, so...


def download_kaggle_data(
    data: dict[str, KaggleRef],
    dst: Path,
    *,
    force: bool = False,
    max_workers: int = 4,
) -> dict[str, Path]:
    if not data:
        return {}

    api = _kaggle_api()
    stop = Event()
    workers = max(1, min(max_workers, len(data)))

    with Progress(
        TextColumn('{task.description}', style='dim'),
        BarColumn(bar_width=30),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console,
        transient=True,
        refresh_per_second=10,
        expand=False,
    ) as progress:
        task_ids = {name: progress.add_task(ref.name, total=None) for name, ref in data.items()}

        def run(name: str) -> Path:
            return _download(api, data[name], dst / name, force, progress, task_ids[name], stop)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run, name): name for name in data}

            try:
                done, _pending = wait(futures, return_when=FIRST_EXCEPTION)
            except KeyboardInterrupt:
                # fast Ctrl-C cancelation niceties
                stop.set()
                console.print('[bold yellow]⊘[/] download interrupted')
                raise SystemExit(130)

            failed = next((f for f in done if f.exception() is not None), None)

            if failed is not None:
                # cancel other ongoing downloads and reraise the error
                stop.set()
                name = futures[failed]
                progress.update(task_ids[name], description=f'{name} failed')
                raise failed.exception()  # ty: ignore

            return {futures[f]: f.result() for f in done}


def require_competition_access(names: Iterable[str]) -> None:

    slugs = competition_slugs(names)
    if not slugs:
        return

    api = _kaggle_api()
    with console.status('Checking Kaggle competition access'):
        with ThreadPoolExecutor(max_workers=min(8, len(slugs))) as pool:
            access = dict(zip(slugs, pool.map(lambda slug: _entered_competition(api, slug), slugs), strict=True))

    missing = [slug for slug, ok in access.items() if not ok]
    if not missing:
        console.print('[bold green]✓[/] competition access verified')
        return

    links = '\n'.join(f'  • [steel_blue3]https://www.kaggle.com/competitions/{slug}[/]' for slug in missing)
    console.print(
        Panel(
            '\nPlease accept the rules for the following Kaggle competitions '
            'and then rerun the command\n'
            '[dim]  Use the account corresponding to the ~/.kaggle/kaggle.json API key[/]'
            '\n\n'
            f'{links}',
            title='[bold]Kaggle competition access required[/]',
            border_style='black',
        )
    )
    raise SystemExit(0)


def _entered_competition(api: KaggleApi, slug: str) -> bool:
    request = ApiListCompetitionsRequest()
    request.group = CompetitionListTab.COMPETITION_LIST_TAB_ENTERED
    request.search = slug
    request.page = 1
    request.page_size = 5

    with api.build_kaggle_client() as client:
        response = client.competitions.competition_api_client.list_competitions(request)

    for competition in response.competitions or []:
        for value in (competition.ref, competition.url):
            if value and value.rstrip('/').split('/')[-1] == slug:
                return True
    return False


# This is a "vendored" KaggleApi.download_file
# it supports restarts, but also does write intermediate files into .part (no correptud)
# plus it is integrated nicer with the Rich progress bar
# original: https://github.com/Kaggle/kaggle-cli/blob/c97b6268ff1c6008207049144037cd556f957db3/src/kaggle/api/kaggle_api_extended.py#L4231


def _download(
    api: KaggleApi,
    file: KaggleRef,
    dst: Path,
    force: bool,
    progress: Progress,
    task_id,
    stop: Event,
) -> Path:
    dst.mkdir(parents=True, exist_ok=True)

    out = dst / (file.local_name or Path(file.name).name)
    part = out.with_name(out.name + '.part')

    if force:
        out.unlink(missing_ok=True)
        part.unlink(missing_ok=True)

    if out.exists():
        _check_existing_file(out, file)
        size = out.stat().st_size
        progress.update(task_id, total=size, completed=size)
        progress.remove_task(task_id)
        return out

    with api.build_kaggle_client() as client:
        if file.kind == 'dataset':
            owner, slug = file.ref.split('/', 1)
            request = ApiDownloadDatasetRequest()
            request.owner_slug = owner
            request.dataset_slug = slug
            request.file_name = file.name
            response = client.datasets.dataset_api_client.download_dataset(request)
        elif file.kind == 'competition':
            request = ApiDownloadDataFilesRequest()
            request.competition_name = file.ref
            response = client.competitions.competition_api_client.download_data_files(request)
        else:
            raise DownloadFailed(f'Unknown Kaggle file kind: {file.kind}')

        _stream_download(
            response,
            out=out,
            part=part,
            file=file,
            progress=progress,
            task_id=task_id,
            stop=stop,
        )

    progress.remove_task(task_id)
    return out


def _stream_download(
    response,
    *,
    out: Path,
    part: Path,
    file: KaggleRef,
    stop: Event,
    progress: Progress,
    task_id,
) -> None:
    content_length = response.headers.get('Content-Length')
    total = int(content_length) if content_length else None
    resumable = response.headers.get('Accept-Ranges') == 'bytes'

    url = response.url
    request = response.request
    method = request.method if request is not None else 'GET'
    headers = dict(request.headers) if request is not None else {}

    response.close()

    for attempt in range(1, _DOWNLOAD_RETRIES + 2):
        if stop.is_set():
            raise DownloadFailed('Interrupted')
        start = part.stat().st_size if resumable and part.exists() else 0

        if total is not None and start >= total:
            break

        request_headers = headers.copy()

        if start:
            request_headers['Range'] = f'bytes={start}-'

        try:
            with requests.request(
                method,
                url,
                headers=request_headers,
                stream=True,
                timeout=_DOWNLOAD_TIMEOUT,
            ) as response:
                response.raise_for_status()
                progress.update(task_id, total=total, completed=start)

                mode = 'ab' if start else 'wb'
                with part.open(mode) as f:
                    for chunk in response.iter_content(_CHUNK_SIZE):
                        if stop.is_set():
                            raise DownloadFailed('Interrupted')

                        if not chunk:
                            continue

                        f.write(chunk)
                        progress.update(task_id, advance=len(chunk))
            break

        except _DOWNLOAD_ERRORS as error:
            if attempt > _DOWNLOAD_RETRIES:
                raise DownloadFailed(f'Could not download {out.name}. Try running the command again.') from error

            if not resumable:
                part.unlink(missing_ok=True)

            if stop.wait(min(2**attempt + random.random(), 60)):
                raise DownloadFailed('Download cancelled') from error

    if total is not None and part.stat().st_size != total:
        part.unlink(missing_ok=True)
        raise DownloadFailed(
            f'Downloaded size for {out.name} does not match Kaggle metadata. Try running the command again.'
        )

    if file.sha256 is not None:
        actual = _sha256(part)
        expected = file.sha256.lower()

        if actual != expected:
            part.unlink(missing_ok=True)
            raise DownloadFailed(f'Checksum mismatch for {out.name}. Try running the command again.')

    part.replace(out)
    size = out.stat().st_size
    progress.update(task_id, total=size, completed=size)


def _check_existing_file(path: Path, file: KaggleRef) -> None:
    if file.sha256 is None:
        return

    actual = _sha256(path)
    expected = file.sha256.lower()

    if actual != expected:
        raise DownloadFailed(
            f'{path.name} already exists, but its checksum does not match. Run again with --force to replace it.'
        )


def _sha256(path: Path) -> str:
    sha = hashlib.sha256()

    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b''):
            sha.update(chunk)

    return sha.hexdigest().lower()
