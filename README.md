# proxmox_lxc_docker_cuda
notes of install and run cuda on docker within lxc and proxmox.

````
+--------------------------------------------------+
|  +--------------------------------------------+  |
|  |  +--------------------------------------+  |  |
|  |  |  +-------------------------------+   |  |  |
|  |  |  |   +---------------------+     |   |  |  |
|  |  |  |   |  Your APP Running   |     |   |  |  |
|  |  |  |   +---------------------+     |   |  |  |
|  |  |  |          CUDA (host)          |   |  |  |
|  |  |  +-------------------------------+   |  |  |
|  |  |               Docker                 |  |  |
|  |  +--------------------------------------+  |  |
|  |               LXC Container                |  |
|  +--------------------------------------------+  |
|            Proxmox Host + CUDA (host)            |
+--------------------------------------------------+

````

Tested with:
- Proxmox version: 8.2.4 (Linux xpve10 6.8.8-3-pve #1 SMP PREEMPT_DYNAMIC PMX 6.8.8-3 (2024-07-16T16:16Z) x86_64 GNU/Linux)
- CPU: Intel 13700KF
- 2x 3090 24GB
- 96GB DDR5

use root for follwing:

**Create a Blacklist Configuration File:**
Driver Conflict: The Nouveau driver is an open-source graphics driver for NVIDIA cards. If it's running, it can conflict with the proprietary NVIDIA driver because both drivers attempt to control the GPU.
   - Open or create the blacklist configuration file using a text editor, for example, `vim`:
     ```bash
     vim /etc/modprobe.d/blacklist-nouveau.conf
     ```
   - Add the following lines to the file:
     ```plaintext
     blacklist nouveau
     options nouveau modeset=0
     ```
   - Reboot
     ```
     update-initramfs -u
     reboot
     ```
  
**PVE Kernel Headers**
  - Install kernel header for all version of pve
   ```
   apt install pve-headers
   ```

**Create udev rules:**
   - Open or create the blacklist configuration file using a text editor, for example, `vim`:
     ```bash
     vim /etc/udev/rules.d/70-nvidia.rules
     ```
   - Add the following lines to the file:
     ```plaintext
     KERNEL=="nvidia", RUN+="/bin/bash -c '/usr/bin/nvidia-smi -L && /bin/chmod 666 /dev/nvidia*'"
     KERNEL=="nvidia_uvm", RUN+="/bin/bash -c '/usr/bin/nvidia-modprobe -c0 -u && /bin/chmod 0666 /dev/nvidia-uvm*'"
     ```
**Download and install cuda 12.5 from official source:**
```
wget https://developer.download.nvidia.com/compute/cuda/12.5.1/local_installers/cuda_12.5.1_555.42.06_linux.run
sh cuda_12.5.1_555.42.06_linux.run
```

**PATH**:
 -   PATH includes /usr/local/cuda-12.5/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-12.5/lib64, or, add /usr/local/cuda-12.5/lib64 to /etc/ld.so.conf and run ldconfig as root
     
**Enable Kernel Modules:**
  - Open: 
  ```bash
       vim /etc/modules-load.d/modules.conf
  ```
  - Add the following lines to the file:
  
  ```plaintext
  # Nvidia modules
  nvidia
  nvidia_uvm
  ```

**Enable LXC conf:**
- Open: 
  ```bash
  vim /etc/pve/lxc/9090.conf
  ```
  - Add the following lines to the file:

  ```
  lxc.cgroup2.devices.allow: c 195:* rwm
  lxc.cgroup2.devices.allow: c 507:* rwm
  lxc.cgroup2.devices.allow: c 243:* rwm
  lxc.mount.entry: /dev/nvidia0 dev/nvidia0 none bind,optional,create=file
  lxc.mount.entry: /dev/nvidia1 dev/nvidia1 none bind,optional,create=file
  lxc.mount.entry: /dev/nvidiactl dev/nvidiactl none bind,optional,create=file
  lxc.mount.entry: /dev/nvidia-uvm dev/nvidia-uvm none bind,optional,create=file
  lxc.mount.entry: /dev/nvidia-uvm-tools dev/nvidia-uvm-tools none bind,optional,create=file
  ```
  also add cpu affinity on multiple cpu system (36 cpus) to reduce memory latency:
  ```
  lxc.cgroup.cpuset.cpus: 0-35
  ```

**Install NV driver and cuda toolkits inside LXC (Ubuntu 22.04):**
Push the cuda file into the container:
```
# enter push cmd
```

Install Driver without kernel header
```
./NVIDIA-Linux-x86_64-5xx.xx.xx.run --no-kernel-modules
```

Install cuda


**Install nvidia container toolkits inside LXC (Ubuntu 22.04):**
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```


**Test inside LXC:**
```
# nvidia-smi

+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.42.06              Driver Version: 555.42.06      CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        On  |   00000000:01:00.0 Off |                  N/A |
|  0%   38C    P8             31W /  370W |       4MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 3090        On  |   00000000:03:00.0 Off |                  N/A |
|  0%   44C    P8             17W /  350W |       4MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

Test Run with gpu hello.cu
```
#include <iostream>
#include <cuda_runtime.h>

__global__ void calculate(int *results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < 20) {
        results[idx] = idx * idx + idx;
    }
}

int main() {
    const int arraySize = 20;
    int results[arraySize];

    // Allocate memory on the GPU
    int *d_results;
    cudaMalloc(&d_results, arraySize * sizeof(int));

    // Define the number of threads and blocks
    int blockSize = 20; // Since arraySize is 20, one block with 20 threads is sufficient
    int numBlocks = 1;  // Only one block is needed

    // Launch the kernel
    calculate<<<numBlocks, blockSize>>>(d_results);

    // Copy the results back to the CPU
    cudaMemcpy(results, d_results, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the results
    for (int i = 0; i < arraySize; ++i) {
        std::cout << i << "*" << i << "+" << i << " = " << results[i] << std::endl;
    }

    // Free GPU memory
    cudaFree(d_results);

    return 0;
}
```
Test it with:
```
root@docker-cuda:~# nvcc test.cu 
root@docker-cuda:~# ./a.out
```

Verify with docker gpu hello world, (go to: https://hub.docker.com/r/nvidia/cuda/tags find a docker you would like to test)
```
docker run --rm --gpus all nvidia/12.5.1-base-ubuntu20.04 nvidia-smi
```
