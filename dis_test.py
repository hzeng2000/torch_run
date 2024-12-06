import os
import torch
def main():
    # Initialize the distributed process group
    torch.distributed.init_process_group(backend="nccl")

    # Get the rank of the current process and the world size (total number of processes)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # Print information about the current process
    print(f"Hello from rank {rank}/{world_size}!")

    # Perform a simple all-reduce operation
    tensor = torch.tensor([rank], dtype=torch.float32, device='cuda')
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)

    # Print the result after the all-reduce operation
    print(f"Rank {rank}: tensor after all-reduce = {tensor.item()}")

if __name__ == "__main__":
    # Ensure the correct device for the current process
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    main()