from typing import List


BOUNDARY_MARKERS = ("\n\n", "\n", ". ", "? ", "! ")


def _find_chunk_end(text: str, start: int, target_end: int, min_end: int) -> int:
    text_length = len(text)
    if target_end >= text_length:
        return text_length

    best_end = -1
    for marker in BOUNDARY_MARKERS:
        marker_index = text.rfind(marker, min_end, target_end)
        if marker_index > best_end:
            best_end = marker_index + len(marker)

    if best_end > start:
        return best_end

    whitespace_index = text.rfind(" ", min_end, target_end)
    if whitespace_index > start:
        return whitespace_index + 1

    return target_end


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[str]:
    """
    Split long text into smaller overlapping chunks.

    Args:
        text: Original document text.
        chunk_size: Maximum number of characters in each chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A list of text chunks.
    """

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        target_end = min(start + chunk_size, text_length)
        min_end = min(start + max(chunk_size // 2, 1), target_end)
        end = _find_chunk_end(text, start, target_end, min_end)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = start + chunk_size - chunk_overlap
        start = max(next_start, 0)

    return chunks
