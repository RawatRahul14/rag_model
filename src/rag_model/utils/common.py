# === Modules ===
import aiofiles
from pathlib import Path


# === Saving Files ===
async def save_files(session_id: str, file_bytes: bytes) -> Path:
    # Create the data directory
    folder_path = Path("data")
    folder_path.mkdir(parents = True, exist_ok = True)

    # Construct the file path
    file_path = folder_path / f"{session_id}.pdf"

    # Save the uploaded bytes
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(file_bytes)

    return file_path