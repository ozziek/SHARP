from src._types import SHARPCard


user_prompt_template = """I need your help with the following document: 

## [{title}]({url}) by {author}

**Highlight:**

I highlighted the following text:
> {highlight}


**Interpretation of the highlight within the context of the document:**
{highlight_interpretation}

## Memory Prompt

{content}

/no_think"""


def format_user_message(row: SHARPCard) -> str:
    source_meta = row["source_meta"]
    assert isinstance(source_meta, dict), "Source meta is not a dictionary"

    assert "author" in source_meta, "Author is not in source meta"
    assert "title" in source_meta, "Title is not in source meta"

    url = row["source_url"]
    assert isinstance(url, str), "URL is not a string"

    # replace newlines with another markdown quote marker
    formatted_highlight = row["highlight"].replace("\n", "\n> ")

    return user_prompt_template.format(
        title=source_meta["title"],
        url=url,
        author=source_meta["author"],
        highlight=formatted_highlight,
        highlight_interpretation=row["highlight_interpretation"],
        content=row["content"],
    )
