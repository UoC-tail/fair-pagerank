"""Automatic downloads of the public dataset files, with checksums."""

import hashlib
import io
import logging
import urllib.request
import warnings
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

_FAIRPRREC = "https://raw.githubusercontent.com/ksemer/fairPRrec/master/Datasets"
_FAIRLAR = (
    "https://media.githubusercontent.com/media/SotirisTsioutsiouliklis/FairLaR/master/Datasets"
)
_PYGDEBIAS = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/NBA"

# Directed graphs in the fairPRrec format: a zip holding out_graph.txt and
# out_community.txt. Blogs and Twitter from fairPRrec (Tsioutsiouliklis et
# al., WWW'22), GitHub from FairLaR (Tsioutsiouliklis et al., WWW'21).
ZIP_URLS = {
    "blogs": f"{_FAIRPRREC}/blogs.zip",
    "twitter": f"{_FAIRPRREC}/twitter.zip",
    "github_female": f"{_FAIRLAR}/github_female.zip",
    "github_male": f"{_FAIRLAR}/github_male.zip",
}

# NBA from the PyGDebias data mirror (raw relationship file, directed).
NBA_URLS = {
    "nba.csv": f"{_PYGDEBIAS}/nba.csv",
    "nba_relationship.txt": f"{_PYGDEBIAS}/nba_relationship.txt",
}

# SHA-256 of the files the paper's results were computed on.
SHA256 = {
    "blogs/out_graph.txt": "1a1d814aff3f26d0b18a2079d0399a26c6bb2e7bb563d8ebdeb69910a161aea8",
    "blogs/out_community.txt": "fc6da5e5ccee3b9bba8b390373ff3abfba8972fd9c7fdd38a516eae90cfee098",
    "twitter/out_graph.txt": "a67f02be80f76c73dda5555c480da9b5532db5ed80d856073b7570692e2f3187",
    "twitter/out_community.txt": "c25b3c6786dbe58aea05261de6eeb42a670bb76eaf7097e3de0140f2f403fe74",
    "github_female/out_graph.txt": (
        "ec6d9240ea44dba0461aa329b64f2b2d7246425845a69059d70a5c2993e59ac5"
    ),
    "github_female/out_community.txt": (
        "4cddb774aceac72a2797e24b84423a5be58eeef013180eb30cc4b12ee2879109"
    ),
    "github_male/out_graph.txt": "36bf07b5a3da8ed2d26a6b7bc6066bbd134dff90d4bfadc6babfbe8d7bd93aa4",
    "github_male/out_community.txt": (
        "27a94e2d74ea5f0d4ba0da7c18255748e68626c2f0582c1889010a262dc670ed"
    ),
    "nba/nba.csv": "fe295195cff01f20b785fe9de1e24062123cb146bdae080ce8b3ea8315d45b80",
    "nba/nba_relationship.txt": "14c73245e5d6b6457cce7504e2bb0ded6b81155e25bf0e7d23793df7c01b4873",
}


def _fetch(url):
    log.info("Downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def _check(path, key):
    expected = SHA256.get(key)
    if expected is None:
        return
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected:
        warnings.warn(
            f"{path} differs from the file used in the paper (sha256 {digest[:12]}... "
            f"!= {expected[:12]}...); results may not match.",
            stacklevel=3,
        )


def ensure_fairprrec_files(name, data_dir, download=True):
    """data_dir/<name>/out_graph.txt and out_community.txt, downloading the
    release zip if they are missing."""
    ddir = Path(data_dir) / name
    files = [ddir / "out_graph.txt", ddir / "out_community.txt"]
    if not all(f.exists() for f in files):
        if not download:
            raise FileNotFoundError(files[0])
        ddir.mkdir(parents=True, exist_ok=True)
        zf = zipfile.ZipFile(io.BytesIO(_fetch(ZIP_URLS[name])))
        for member in zf.namelist():
            base = Path(member).name
            if base in ("out_graph.txt", "out_community.txt"):
                (ddir / base).write_bytes(zf.read(member))
    for f in files:
        _check(f, f"{name}/{f.name}")
    return files


def ensure_nba_files(data_dir, download=True):
    """data_dir/nba/nba.csv and nba_relationship.txt from the PyGDebias mirror."""
    ddir = Path(data_dir) / "nba"
    paths = []
    for fname, url in NBA_URLS.items():
        path = ddir / fname
        if not path.exists():
            if not download:
                raise FileNotFoundError(path)
            ddir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_fetch(url))
        _check(path, f"nba/{fname}")
        paths.append(path)
    return paths
