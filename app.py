def flexible_match(text: str, term: str) -> bool:
    """
    Return True if the term is found in text, matching exactly (case-insensitive).
    For example, if term is "Chamber", it will match "Chamber" but not "Chambers".
    """
    pattern = r'\b' + re.escape(term) + r'\b'
    return re.search(pattern, text, re.IGNORECASE) is not None

def extract_snippet(text: str, search_term: str, context: int = 50) -> str:
    """
    Extract a snippet with up to 'context' characters before and after the search term.
    Uses an exact (case-insensitive) word-boundary match for the search term.
    """
    pattern = r"(?i)(.{{0,{}}}\b{}\b.{{0,{}}})".format(context, re.escape(search_term), context)
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return text[:200].strip()  # fallback: first 200 characters

def extract_exact_mentions(chunks: List[Dict[str, Any]], search_term: str) -> List[Dict[str, Any]]:
    """
    For each chunk that contains the exact search term (using word-boundary matching),
    extract one snippet per file and page.
    This groups the results by (file, page) so that if a page was split into multiple chunks,
    only one snippet is returned.
    """
    results = {}
    for chunk in chunks:
        original_text = chunk.get("text", "")
        # Normalize whitespace
        text_norm = ' '.join(original_text.split())
        if flexible_match(text_norm, search_term):
            snippet = extract_snippet(text_norm, search_term)
            file_name = chunk["source"].get("file", "unknown file")
            page = chunk["source"].get("page")
            if page:
                page = str(page).strip()
                m = re.search(r'\d+', page)
                page = m.group(0) if m else "N/A"
            else:
                page = "N/A"
            key = (file_name, page)
            if key not in results:
                results[key] = {
                    "file": file_name,
                    "page": page,
                    "snippet": snippet
                }
    # Return the results as a sorted list (by file then by page number)
    sorted_results = sorted(
        results.values(),
        key=lambda x: (x["file"].lower(), int(x["page"]) if x["page"].isdigit() else 0)
    )
    return sorted_results

