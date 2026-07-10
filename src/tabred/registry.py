from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

_TABRED_KAGGLE_DATASET = 'irubachev/tabred'

KaggleKind = Literal['dataset', 'competition']


@dataclass(frozen=True)
class KaggleRef:
    kind: KaggleKind
    ref: str
    name: str
    sha256: str | None
    local_name: str | None = None

    @property
    def slug(self):
        return self.ref if self.kind == 'competition' else None


@dataclass(frozen=True)
class Dataset:
    name: str

    raw: KaggleRef
    preprocessed: KaggleRef


def dataset_names(names: Iterable[str]) -> list[str]:
    names = list(names)
    result = list(DATASETS) if names == ['all'] else names
    unknown = sorted(set(result) - set(DATASETS))
    if unknown:
        raise SystemExit('Unknown dataset(s): ' + ', '.join(unknown) + '. Available datasets: ' + ', '.join(DATASETS))
    return result


def competition_slugs(names: Iterable[str]) -> list[str]:
    return [ds.raw.slug for name in dataset_names(names) if (ds := DATASETS[name]).raw.kind == 'competition']


def get_raw_files(names: Iterable[str]) -> dict[str, KaggleRef]:
    return {name: DATASETS[name].raw for name in dataset_names(names)}


def get_preprocessed_files(names: Iterable[str]) -> dict[str, KaggleRef]:
    return {name: DATASETS[name].preprocessed for name in dataset_names(names)}


def _preprocessed_ref(name: str) -> KaggleRef:
    return KaggleRef(
        'dataset',
        _TABRED_KAGGLE_DATASET,
        f'preprocessed/{name}.tabred',
        _PREPROCESSED_SHA256[name],
    )


def _competition_dataset(name, slug):
    return Dataset(
        name=name,
        raw=KaggleRef('competition', slug, f'{slug}.zip', None),
        preprocessed=_preprocessed_ref(name),
    )


def _yandex_dataset(name, filename):
    return Dataset(
        name=name,
        raw=KaggleRef(
            'dataset',
            _TABRED_KAGGLE_DATASET,
            f'raw/{name}/{filename}',
            _RAW_SHA256[name],
            local_name=filename,
        ),
        preprocessed=_preprocessed_ref(name),
    )


_PREPROCESSED_SHA256 = {
    'cooking-time': 'ff7fb7f5be7101019164ec190831766997c28b00aefb4b902050099948435d95',
    'delivery-eta': '097b1272e45de3a723f66d7b0d5c0a397b53a773cd5a781b8f094bfb753d5362',
    'ecom-offers': 'de7f96400b9006eab4bfa4318993506aa667c0508da32a5e00b912b2bc702bc9',
    'homecredit-default': 'fdcdcfe9b67e9ac8fae0291ee0aa8b492e41c123f99c9d3cecae2b78602c7f30',
    'homesite-insurance': '9c94a9ba8a2dc68221c97d45273076714ad308b5e556ac59fd52117e1ea17d75',
    'maps-routing': 'bdf503d49f10af2594594feb8c2ad8f4d90725bf21f1fe3de238acf0cb9f2abf',
    'sberbank-housing': 'a6a11bc09204a5d6d0015dfae504cd630186c813a87e384eaf8caf5b960eb334',
    'weather': 'b5dfc96c4e33d3567b748f0cc2b12baa5ea28fac400147953a3cd1cb735697e5',
}

_RAW_SHA256 = {
    'cooking-time': '330b0b811195e7a6b75a43a7bd0ab65b52c83967c8f24d4765ccd2584c3228b9',
    'delivery-eta': '37dc8c9479821da3539d1c42f6d97afbc4bf8b1466604839edac834ceb9e1ef5',
    'maps-routing': '225712c539cf18503347c2d88bf4e4e9f624f180e7156a0fc4fedb19c43775b2',
    'weather': '1984681822f14d24fbb491a8935446f9a263f48a469564c038a856b8194195e8',
}


DATASETS = {
    name: _yandex_dataset(name, filename)
    for name, filename in [
        ('cooking-time', 'cooking_time.parquet'),
        ('delivery-eta', 'delivery_eta.parquet'),
        ('maps-routing', 'maps_routing.parquet'),
        ('weather', 'weather.parquet'),
    ]
} | {
    name: _competition_dataset(name, slug)
    for name, slug in [
        ('sberbank-housing', 'sberbank-russian-housing-market'),
        ('ecom-offers', 'acquire-valued-shoppers-challenge'),
        ('homesite-insurance', 'homesite-quote-conversion'),
        ('homecredit-default', 'home-credit-credit-risk-model-stability'),
    ]
}
