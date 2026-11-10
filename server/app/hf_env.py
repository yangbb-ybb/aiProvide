"""模型下载的运行时环境准备（下载镜像 + 缓存目录）。

两个环境变量都必须在 **import fastembed 之前**设好：

1. `HF_ENDPOINT` —— fastembed 的模型权重托管在 huggingface.co，
   国内直连基本会超时挂住。hf-mirror 是官方国内镜像，文件是同一份。
   huggingface_hub 在 **import 时**就把 HF_ENDPOINT 读进自己的常量，
   之后再改 os.environ 不生效。

2. `FASTEMBED_CACHE_PATH` —— fastembed 默认把权重缓存到
   `tempfile.gettempdir()/fastembed_cache`，也就是 macOS 的 /var/folders/.../T/，
   **重启或磁盘紧张时会被系统清掉**，1.7GB 的模型得重下。改成 ~/.cache/fastembed。

优先级都是：系统环境变量 > server/.env > 默认值。
这样临时用 `FASTEMBED_CACHE_PATH=xxx uv run ...` 就能覆盖，不用改文件。
"""

from __future__ import annotations

import os
from pathlib import Path

# huggingface.co 直连超时时用的镜像
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"

# 模型缓存目录（放在用户主目录，系统不会清理）
DEFAULT_FASTEMBED_CACHE = Path.home() / ".cache" / "fastembed"


def configure_hf_endpoint(endpoint: str | None = None) -> str:
    """设置 HF_ENDPOINT，返回最终生效的地址。"""
    target = os.environ.get("HF_ENDPOINT") or endpoint or DEFAULT_HF_ENDPOINT
    os.environ["HF_ENDPOINT"] = target
    return target


def configure_fastembed_cache(cache_path: str | Path | None = None) -> str:
    """设置 FASTEMBED_CACHE_PATH，返回最终生效的目录。

    支持 `~` 开头的路径（.env 里写起来比绝对路径方便）。
    """
    raw = (
        os.environ.get("FASTEMBED_CACHE_PATH")
        or (str(cache_path) if cache_path else None)
        or str(DEFAULT_FASTEMBED_CACHE)
    )
    target = str(Path(raw).expanduser())
    os.environ["FASTEMBED_CACHE_PATH"] = target
    Path(target).mkdir(parents=True, exist_ok=True)
    return target
