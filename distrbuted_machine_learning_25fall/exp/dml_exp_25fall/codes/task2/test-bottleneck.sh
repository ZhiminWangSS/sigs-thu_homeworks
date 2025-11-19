#!/bin/bash

# # Test script for bottleneck scenarios
# echo "Testing bottleneck scenarios..."

# # Test 1: No bottleneck (baseline)
# echo "=== Test 1: No bottleneck (baseline) ==="
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./model-bottleneck.py --n_devices 4 --comm_method allreduce --bottleneck_type none

# # Test 2: Time delay bottleneck on rank 0
# echo "=== Test 2: Time delay bottleneck on rank 0 ==="
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./model-bottleneck.py --n_devices 4 --comm_method allreduce --bottleneck_type time_delay --bottleneck_rank 0 --sleep_time 0.1

# # Test 3: Time delay bottleneck on rank 2
echo "=== Test 3: Time delay bottleneck on rank 2 ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./model-bottleneck.py --n_devices 4 --comm_method allreduce --bottleneck_type time_delay --bottleneck_rank 2 --sleep_time 0.001

# Test 4: Uneven batch size bottleneck on rank 1
# echo "=== Test 4: Uneven batch size bottleneck on rank 1 ==="
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./model-bottleneck.py --n_devices 4 --comm_method allreduce --bottleneck_type uneven_batch --bottleneck_rank 1

# # Test 5: Different communication methods with bottleneck
# echo "=== Test 5: AllReduce with time delay ==="
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./model-bottleneck.py --n_devices 4 --comm_method allreduce --bottleneck_type time_delay --bottleneck_rank 0 --sleep_time 0.1

# echo "=== Test 6: AllGather with time delay ==="
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./model-bottleneck.py --n_devices 4 --comm_method allgather --bottleneck_type time_delay --bottleneck_rank 0 --sleep_time 0.1

# echo "=== Test 7: Reduce with time delay ==="
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./model-bottleneck.py --n_devices 4 --comm_method reduce --bottleneck_type time_delay --bottleneck_rank 0 --sleep_time 0.1

echo "All tests completed! Check TensorBoard logs in ./runs/ directory for performance analysis."