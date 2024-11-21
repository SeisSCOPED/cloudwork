# CloudWork: Cloud-Native Earthquake Detection and Picking

**CloudWork** is a Python-based software that uses **[SeisBench](https://github.com/seisbench/seisbench)**-compatible machine learning workflows to detect, pick, and associate earthquakes. This software is tailored for **cloud-native workflows on AWS** and serves as an exercise for processing seismic data from **NCEDC**, **SCEDC**, and **EarthScope**, all hosted on S3.

---

## Features

- **SeisBench Compatibility**: Leverage SeisBench's machine learning models for robust earthquake detection, picking, and association.
- **Cloud-Native Workflow**: Optimized for deployment on AWS with support for serverless computing and scalable processing.
- **AWS Data Integration**: Seamless access to seismic data from NCEDC, SCEDC, and EarthScope stored on S3.
- **Customizable & Extendable**: Designed as an exercise and foundation for building advanced seismic analysis pipelines.

---

## Installation

### 1. Using Conda Environment

To set up the Python environment for CloudWork:

1. Clone the repository:

   ```bash
   git clone https://github.com/SeisSCOPED/cloudwork.git
   cd cloudwork
   ```

2. Create and activate a conda environment:

   ```bash
   conda create --name scoped python=3.12
   conda activate scoped
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### 2. Using Docker

To run the software in a containerized environment:

1. Build the Docker image:

   ```bash
   docker build -t cloudwork .
   ```

2. Run the container:

   ```bash
   docker run -v $(pwd):/app -it cloudwork
   ```

This setup ensures all dependencies are properly isolated and compatible across systems.

---

## AWS Batch Integration

CloudWork is designed to run efficiently on AWS Batch, enabling scalable processing of large seismic datasets. Follow [this tutorial on AWS Batch with SeisBench](https://seisscoped.org/HPS-book/chapters/quake_catalog/seisbench_catalog.html) to set up and deploy CloudWork in an AWS environment.

---

## Usage

### Basic Workflow

1. **Prepare Input Data**: Organize seismic data from S3-compatible buckets.
2. **Configure Parameters**: Update configuration files in the repository for model settings and data access.
3. **Run CloudWork**: Execute the workflow either locally or in your AWS environment.

See the full AWS-batch tutorial on the [SCOPED HPS Book](https://seisscoped.org/HPS-book/chapters/quake_catalog/seisbench_catalog.html).

---

## Citation

If you use **CloudWork** in your research, please cite the repository as follows (pending publication of associated paper):

> **CloudWork**: A cloud-native Python software for earthquake detection, picking, and association. Developed by SeisSCOPED contributors Jannes Munchmeyer, Yiyu Ni, Marine Denolle. Available at: [https://github.com/SeisSCOPED/cloudwork](https://github.com/SeisSCOPED/cloudwork).


---

## Contributing

Contributions are welcome! If you'd like to contribute to CloudWork:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request.

For detailed contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Support

For issues or questions, please open an issue in the [GitHub repository](https://github.com/SeisSCOPED/cloudwork/issues).

---

This README incorporates best practices, clear installation instructions, AWS integration details, and citation guidance, making it comprehensive and user-friendly.
