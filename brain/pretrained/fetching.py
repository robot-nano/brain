import urllib.request
import urllib.error
import pathlib
import logging
import huggingface_hub
from requests.exceptions import HTTPError

logger = logging.getLogger(__name__)


def _missing_ok_unlink(path):
    # missing_ok=True was added to Path.unlink() in Python 3.8
    # This does the same.
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def fetch(
    filename,
    source,
    savedir="./pretrained_model_checkpoints",
    overwrite=False,
    save_filename=None,
    use_auth_token=False,
    revision=None,
):
    if save_filename is None:
        save_filename = filename
    savedir = pathlib.Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)
    sourcefile = f"{source}/{filename}"
    destination = savedir / save_filename
    if destination.exists() and not overwrite:
        MSG = f"Fetch {filename}: Using existing file/symlink in {str(destination)}."
        logger.info(MSG)
        return destination
    if str(source).startswith("http:") or str(source).startswith("https:"):
        # Interpret source as web address.
        MSG = (
            f"Fetch {filename}: Downloading from normal URL {str(sourcefile)}."
        )
        logger.info(MSG)
        # Download
        try:
            urllib.request.urlretrieve(sourcefile, destination)
        except urllib.error.URLError:
            raise ValueError(
                f"Interpreted {source} as web address, but could not download."
            )
    elif pathlib.Path(source).is_dir():
        # Interpret source as local directory path
        # Just symlink
        sourcepath = pathlib.Path(sourcefile).absolute()
        MSG = f"Fetch {filename}: Linking to local file in {str(sourcepath)}."
        logger.info(MSG)
        _missing_ok_unlink(destination)
        destination.symlink_to(sourcepath)
    else:
        # Interpret source as huggingface hub ID
        # Use huggingface hub's fancy cached download.
        MSG = f"Fetch {filename}: Delegating to Huggingface hub, source {str(source)}."
        logger.info(MSG)
        try:
            fetched_file = huggingface_hub.hf_hub_download(
                repo_id=source,
                filename=filename,
                use_auth_token=use_auth_token,
                revision=revision,
            )
        except HTTPError as e:
            if "404 Client Error" in str(e):
                raise ValueError("File not found on HF hub")
            else:
                raise

        # Huggingface hub downloads to etag filename, symlink to the expected one:
        sourcepath = pathlib.Path(fetched_file).absolute()
        _missing_ok_unlink(destination)
        destination.symlink_to(sourcepath)
    return destination
