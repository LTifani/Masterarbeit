# Masterarbeit

# Prompt

> Improve the Python code (see attachment) to make it more professional, readable, and efficient.
> Requirements:
>
> * Use clear, consistent, and descriptive variable and function names.
> * Optimize the implementation for performance and maintainability.
> * Follow Python best practices (PEP 8 and clean code principles).
> * Write all docstrings and inline comments **in English**, using the Google or NumPy docstring style.
> * Ensure logical structure, modularity, and readability.
> * Do not change the functionality, only improve clarity and quality.

# Combine all script for LLM analyse

- Generate the file:
    ```shell
    find . -type f -exec sh -c 'echo "=== {} ==="; cat "{}"; echo -e "\n\n"' \; > ../output/combined_output.txt
    ```
- Copy the file to windows PC
    ```shell
    scp kommegle@lkommegne.ai.tha.de:~/Masterarbeit/output/combined_output.txt .
    ```

# Generate requirement file

```shell
    pip install pipreqs
```