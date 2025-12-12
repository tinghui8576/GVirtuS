# GVirtuS Kubernates Integration
GVirtuS follows the architecture:
- Backend: Runs on nodes with physical GPU hardware.
- Frontend: Can run anywhere and offloads GPU calls to the backend.

> [!IMPORTANT]
> To integrate GVirtuS with Kubernetes, make sure you have one node that contains physical GPU in your cluster.

## Setup
Create the `gvirtus-system` namespace:
```
kubectl apply -f k8s/gvirtus-ns
```

Select the node contains GPU as the one you want to serve as GVirtuS backends.
```
kubectl label node <node-name> nvidia.com/gpu.present=true
```

Create the `gvirtus-backend` daemonset and service:
```
kubectl apply -f k8s/gvirtus-backend
```

The node labeled with `nvidia.com/gpu.present` will run a pod that serves as the GVirtuS backend. You can verify that the GVirtuS backend pod is running on each node with the following command.
```
kubectl get pods -n gvirtus-system -l app.kubernetes.io/component=backend -o wide
```

## Example Execution
After you have successfully deployed one node runs the gvirtus backend, you are ready to run a gvirtus application.
```
kubectl apply -f k8s/gvirtus-example-app/
```

If everything is set up properly and gvirtus backend service is running on the cluster, the application should successfully be scheduled and executed normally.
You can check the log for both frontend example
```
kubectl logs gvirtus-frontend-pod -n gvirtus-system
```
You should see logs similar to the following:
```bash
...
DEBUG - DEBUG - Received routine cudaDeviceSynchronize [pid=1, tid=1] 
DEBUG - Write 0 bytes to the buffer
DEBUG - Read 0 bytes from the buffer
DEBUG - Routine 'cudaDeviceSynchronize' returned 0 | server_exec=0s | send=0s | recv=0s | in=0B | out=0B | pid=1 tid=1
DEBUG - DEBUG - Called: cudaDeviceSynchronize
DEBUG - DEBUG - Received routine cudaUnregisterFatBinary [pid=1, tid=1]
DEBUG - Write 31 bytes to the buffer
DEBUG - Read 0 bytes from the buffer
DEBUG - Routine 'cudaUnregisterFatBinary' returned 0 | server_exec=0s | send=0s | recv=0s | in=31B | out=0B | pid=1 tid=1
DEBUG - DEBUG - Called: cudaUnregisterFatBinary
```
