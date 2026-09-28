Act as an OCR Model. Extract the exact details of the image into a markdown format.

## Output:
````markdown
````

With this setup, `from prompts import BASELINE_OCR_PROMPT` works as before, and you can edit any prompt without touching Python code.

If a prompt needs values inserted (like a JSON schema), use `.replace("{schema}", value)` rather than `.format()`. `.format()` breaks on the `{ }` braces inside JSON examples.