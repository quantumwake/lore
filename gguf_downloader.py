"""Custom GGUF downloader with real progress tracking."""

import os
import requests
from pathlib import Path
from typing import Optional, Callable
from urllib.parse import urlparse
import hashlib
import shutil
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
    TaskProgressColumn
)

console = Console()


def download_with_progress(url: str,
                          destination: str,
                          chunk_size: int = 8192 * 1024,  # 8MB chunks
                          resume: bool = True) -> str:
    """Download a file with progress bar support."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists and get current size
    file_size_existing = 0
    if destination.exists() and resume:
        file_size_existing = destination.stat().st_size

    # Get file info
    headers = {}
    if file_size_existing > 0 and resume:
        headers['Range'] = f'bytes={file_size_existing}-'

    response = requests.get(url, headers=headers, stream=True, allow_redirects=True)

    # Check if server supports resume
    if resume and file_size_existing > 0:
        if response.status_code == 416:  # Range not satisfiable = file complete
            console.print(f"[green]✓ File already downloaded: {destination.name}[/green]")
            return str(destination)
        elif response.status_code != 206:  # Partial content not supported
            file_size_existing = 0
            response = requests.get(url, stream=True, allow_redirects=True)

    response.raise_for_status()

    # Get total file size
    total_size = int(response.headers.get('content-length', 0))
    if file_size_existing > 0:
        total_size += file_size_existing

    # Setup progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TextColumn("•"),
        DownloadColumn(),
        TextColumn("•"),
        TransferSpeedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=2
    ) as progress:

        task = progress.add_task(
            f"[cyan]{destination.name}[/cyan]",
            total=total_size,
            completed=file_size_existing
        )

        # Download file
        mode = 'ab' if file_size_existing > 0 else 'wb'

        with open(destination, mode) as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))

    console.print(f"[green]✓ Download complete: {destination.name}[/green]")
    return str(destination)


def download_gguf_direct(repo_id: str,
                        filename: str,
                        cache_dir: str = "~/.cache/ubercli/models",
                        token: Optional[str] = None) -> str:
    """Download GGUF model directly from HuggingFace with progress."""

    cache_dir = Path(os.path.expanduser(cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build download URL
    base_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"

    # Add token if provided
    if token:
        base_url += f"?token={token}"

    # Create destination path
    repo_name = repo_id.replace("/", "_")
    model_dir = cache_dir / repo_name
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / filename

    # Check if already downloaded
    if destination.exists():
        # Verify file size
        response = requests.head(base_url, allow_redirects=True)
        expected_size = int(response.headers.get('content-length', 0))
        actual_size = destination.stat().st_size

        if actual_size == expected_size:
            console.print(f"[green]✓ Model already cached: {filename} ({actual_size / (1024**3):.2f} GB)[/green]")
            return str(destination)
        else:
            console.print(f"[yellow]Partial download found ({actual_size / (1024**3):.2f} GB of {expected_size / (1024**3):.2f} GB), resuming...[/yellow]")

    console.print(f"\n[cyan]Downloading {filename} from {repo_id}...[/cyan]")

    # Get file size first
    response = requests.head(base_url, allow_redirects=True)
    file_size = int(response.headers.get('content-length', 0))
    if file_size:
        console.print(f"[dim]File size: {file_size / (1024**3):.2f} GB[/dim]\n")

    # Download with progress
    return download_with_progress(base_url, str(destination), resume=True)