#!bin/bash


NODES=("1" "3" "4")

CHUNK_PREFIX="chunk_"
USER="$USER"
WORK_DIR="homework1"
DATA_FILE="large_wc_dataset.txt"


# 切分文件：按行数均分（简单起见）
total_lines=$(wc -l < "$DATA_FILE")
lines_per_chunk=$((total_lines / ${#NODES[@]} + 1))
echo "Total lines: $total_lines"
echo "Lines per chunk: $lines_per_chunk"


#
split -l "$lines_per_chunk" "$DATA_FILE" "$CHUNK_PREFIX"

# 获取实际分片列表
chunks=(${CHUNK_PREFIX}*)
num_chunks=${#chunks[@]}
echo "Created $num_chunks chunks"

# 分发分片到各节点
for i in "${!NODES[@]}"; do
    node=${NODES[$i]}
    chunk=${chunks[$i]}
#    if [ -z "$chunk" ]; then
#        echo "Warning: No chunk for $node"
#        continue
#    fi
    echo "Sending $chunk to $node"
    ssh "thumm0${NODES[$i]}" "mkdir -p $WORK_DIR"
    scp "$chunk" "thumm0${NODES[$i]}:~/$WORK_DIR/"
    scp count_words.sh "thumm0${NODES[$i]}:~/$WORK_DIR/"
done

# 并行执行统计
echo "Starting distributed counting..."
start_time=$(date +%s.%N)

for i in "${!NODES[@]}"; do
    node=${NODES[$i]}
    chunk=${chunks[$i]}
    if [ -z "$chunk" ]; then continue; fi
    echo "Running on $node for $chunk"
    ssh "thumm0${NODES[$i]}" "cd "$WORK_DIR" && bash count_words.sh $chunk > result_node.txt" &
done

# ================== 汇总各节点结果 ==================
echo "Collecting results from all nodes..."

# 清理本地旧结果
rm -f result_thumm0*.txt

# 从每个节点拉取结果
for i in "${!NODES[@]}"; do
    node="thumm0${NODES[$i]}"
    chunk="${chunks[$i]}"
    if [ -z "$chunk" ]; then continue; fi

    echo "Fetching result from $node"
    scp "$node:$WORK_DIR/result_node.txt" "./result_${node}.txt"
done

# 使用 awk 合并所有 result_thumm0*.txt


awk '
{
    total[$1] += $3
}
END {
    for (i = 65; i <= 90; i++) {
        c = sprintf("%c", i)
        print c " 频次: " (total[c] + 0)
    }
    for (i = 97; i <= 122; i++) {
        c = sprintf("%c", i)
        print c " 频次: " (total[c] + 0)
    }
}' result_thumm0*.txt > total_result.txt


echo "✅ Final result saved too total_result.txt"
