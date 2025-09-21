#!/usr/bin/env python3
"""Test HuggingFace download with proper progress tracking."""

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import tqdm as hf_tqdm
from rich.progress import (
    Progress,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
    TextColumn,
    SpinnerColumn,
    MofNCompleteColumn
)
from rich.console import Console
import sys

console = Console()

# Store the original tqdm
original_tqdm = hf_tqdm.tqdm

# Global progress object
rich_progress = None
progress_task = None

class TqdmToRich:
    """Wrapper to redirect tqdm progress to rich progress bar."""

    def __init__(self, *args, **kwargs):
        self.total = kwargs.get('total', None)
        self.desc = kwargs.get('desc', '')
        self.n = 0
        self.disable = kwargs.get('disable', False)

        global rich_progress, progress_task

        if not self.disable and rich_progress and self.total:
            # Update the task total
            rich_progress.update(progress_task, total=self.total)

    def update(self, n=1):
        """Update progress."""
        self.n += n
        global rich_progress, progress_task
        if not self.disable and rich_progress and progress_task is not None:
            rich_progress.update(progress_task, completed=self.n)

    def close(self):
        """Close progress bar."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

def download_with_rich_progress(repo_id: str, filename: str):
    """Download with Rich progress bar."""

    global rich_progress, progress_task

    # Create Rich progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        DownloadColumn(),
        TextColumn("•"),
        TransferSpeedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=2
    ) as progress:

        rich_progress = progress
        progress_task = progress.add_task(
            f"[cyan]Downloading {filename}[/cyan]",
            total=None
        )

        # Monkey-patch tqdm in huggingface_hub
        import huggingface_hub.utils.tqdm as hf_tqdm_module
        hf_tqdm_module.tqdm = TqdmToRich

        # Also patch the main huggingface_hub.file_download module
        import huggingface_hub.file_download as fd
        fd.tqdm = TqdmToRich

        try:
            # Download the file
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir="~/.cache/test",
                resume_download=True
            )

            console.print(f"\n[green]✓ Downloaded to: {path}[/green]")
            return path

        finally:
            # Restore original tqdm
            hf_tqdm_module.tqdm = original_tqdm
            fd.tqdm = original_tqdm
            rich_progress = None
            progress_task = None

if __name__ == "__main__":
    # Test with a small model first
    repo = "TheBloke/Llama-2-7B-GGUF"
    file = "llama-2-7b.Q2_K.gguf"  # Small 2.8GB file for testing

    console.print(f"[cyan]Testing download of {file} from {repo}[/cyan]\n")
    download_with_rich_progress(repo, file)