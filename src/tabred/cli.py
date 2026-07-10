from pathlib import Path

import cyclopts

import tabred

app = cyclopts.App(name='tabred')


@app.command
def download(
    names: list[str] = ['all'],
    *,
    output_path: Path = Path('data'),
    force: bool = False,
    max_workers: int = 4,
) -> None:
    tabred.download_preprocessed_data(
        names,
        output_path,
        force=force,
        max_workers=max_workers,
    )


def main() -> None:
    app()
