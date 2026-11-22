"""Fitur yang dihitung ulang langsung dari string URL, penanda ketidakcocokan dengan fitur bawaan, dan tipe URL."""
from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse

import numpy as np
import pandas as pd

RISKY_EXT = r"\.(exe|zip|rar|sh|apk|bin|dll|jar|scr|war|m|a|i|arm\d*|mips|x86)$"


def entropy(s: str) -> float:
    c = Counter(s)
    n = len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def _host(u: str) -> str:
    try:
        return urlparse(u if "://" in u else "http://" + u).hostname or ""
    except ValueError:
        return ""


def url_type(urls: pd.Series) -> pd.Series:
    """Tiga tipe URL: host berupa alamat IP, beranda polos https://www.<domain>, dan lainnya."""
    host = urls.map(_host)
    ip = host.str.fullmatch(r"[\d.]+").fillna(False)
    home = urls.str.fullmatch(r"https://www\.[^/?#]+/?").fillna(False)
    return pd.Series(np.select([ip, home], ["host IP", "beranda https://www"], "lainnya"), index=urls.index)


def recompute(urls: pd.Series) -> pd.DataFrame:
    rows = []
    for u in urls:
        p = urlparse(u if "://" in u else "http://" + u)
        try:
            port = p.port is not None
        except ValueError:
            port = False
        host = p.hostname or ""
        parts = [x for x in host.split(".") if x]
        rows.append({
            "u_len": len(u), "u_https": int(u.startswith("https://")), "u_www": int(host.startswith("www.")),
            "u_ip": int(bool(re.fullmatch(r"[\d.]+", host))), "u_port": int(port), "u_host_len": len(host),
            "u_n_labels": len(parts), "u_tld_len": len(parts[-1]) if parts else 0, "u_hyphen": host.count("-"),
            "u_path_len": len(p.path), "u_path_depth": p.path.count("/"), "u_query_len": len(p.query),
            "u_n_params": len([q for q in p.query.split("&") if q]), "u_digits": sum(ch.isdigit() for ch in u),
            "u_upper": sum(ch.isupper() for ch in u), "u_special": sum(not ch.isalnum() for ch in u),
            "u_entropy": entropy(u), "u_risky_ext": int(bool(re.search(RISKY_EXT, p.path, re.I))),
            "u_tokens": len([t for t in re.split(r"[/.?=&\-_:]+", u) if t]),
        })
    out = pd.DataFrame(rows, index=urls.index)
    out["u_pct_digits"] = out.u_digits / out.u_len * 100
    return out


def mismatch(provided: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """1 jika fitur bawaan tidak sama dengan nilai yang dihitung ulang dari URL."""
    return pd.DataFrame({
        "beda_entropy": ~np.isclose(provided.url_entropy, clean.u_entropy, atol=1e-3),
        "beda_path_length": ~np.isclose(provided.path_length, clean.u_path_len, atol=1e-2),
        "beda_persen_digit": ~np.isclose(provided.percentage_numeric_chars, clean.u_pct_digits, atol=1e-2),
        "token_count_pecahan": provided.token_count != provided.token_count.round(),
    }, index=provided.index).astype(int)
