# !bin/bash

# 检查是否提供了文件参数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

FILE_DIR="$1"

# 检查文件是否存在
if [ ! -f "$FILE_DIR" ]; then
    echo "Error: File '$FILE_DIR' does not exist."
    exit 1
fi

for c in {A..Z} {a..z}; do
    n=$(grep -o "$c" "$FILE_DIR" | wc -l)
    echo "$c 频次: $n"
done

