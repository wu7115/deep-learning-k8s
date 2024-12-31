# KubeML: Distributed Deep Learning Training on Kubernetes

This project demonstrates how to train multiple deep neural networks simultaneously using Kubernetes for orchestration. Currently implemented with Minikube for local testing, with plans to scale to cloud platforms like AWS EKS.

## Models

- **CNN (Convolutional Neural Network)**: Optimized for spatial feature learning
- **LSTM (Long Short-Term Memory)**: Processes sequential aspects of image data

## Prerequisites

- Docker
- Minikube
- kubectl
- Python 3.9+
- TensorFlow 2.x

## Getting Started

1. Start Minikube:
```bash
minikube start
```

2. Build Docker images:
```bash
# Build CNN image
cd cnn
docker build -t {your cnn image name} .

# Build LSTM image
cd ../lstm
docker build -t {your lstm image name} .
```

3. Push onto docker hub:
```bash
docker push {your cnn image name}
docker push {your lstm image name}
```

4. Deploy models:
```bash
kubectl apply -f kubernetes/cnn-deployment.yaml
kubectl apply -f kubernetes/lstm-deployment.yaml
```

5. Monitor training:
```bash
kubectl get pods
kubectl logs -f deployment/cnn-deployment
kubectl logs -f deployment/lstm-deployment
```

## Resource Management

Both models are configured with resource limits:
- Memory: 2Gi (request) / 4Gi (limit)
- CPU: 1 (request) / 2 (limit)

These can be adjusted in the deployment YAML files based on available resources.

## Future Work

### AWS EKS Integration
- Deploy to AWS EKS for production-scale training
- Leverage multiple instances for true parallel training
- Implement auto-scaling based on workload

### Planned Features
- Multiple GPU support
- Distributed data loading
- Model checkpointing and recovery
- Training progress monitoring dashboard
- Dynamic resource allocation
- Integration with MLflow for experiment tracking

## Contributing

Feel free to open issues or submit pull requests. All contributions are welcome!

## License

MIT License

## Authors

- Daniel Wu

## Acknowledgments

- TensorFlow team for the deep learning framework
- Kubernetes community for container orchestration
