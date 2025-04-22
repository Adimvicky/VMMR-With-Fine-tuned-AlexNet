# Vehicle Make and Model Recognition with Fine-Tuned AlexNet

This project explores the effectiveness of fine-tuning AlexNet for vehicle make and model recognition using the Stanford Cars dataset. All training and evaluations were performed using `project_colab.ipynb` on Google Colab with GPU acceleration (A100). The final report is included in `report.pdf`.

## 📁 Project Structure

Group09_Project/ ├── dataset/ # Stanford Cars dataset (organized into train and valid) │ ├── train/ │ └── valid/ ├── models/ │ └── best_alexnet.pkl # Saved best-performing AlexNet model after fine-tuning ├── visualizations/ # Evaluation plots and CSVs │ ├── confusion_matrix.png # Raw confusion matrix │ ├── confusion_matrix_normalized.png # Normalized confusion matrix │ ├── lr_find_plot.png # Learning rate finder plot │ ├── lr_find_plot_labeled.png # LR plot with valley marked │ ├── sample_batch.png # Sample training images │ ├── sample_predictions.png # Grid of predicted vs actual labels │ ├── top_losses.png # High-loss predictions │ ├── top_losses_examples.png # More misclassified samples │ ├── top_losses_per_class.png # Top errors across different classes │ ├── train_loss.png # Training loss only │ ├── train_vs_val_loss.png # Combined train and validation loss │ ├── val_accuracy_over_epochs.png # Accuracy progression over epochs │ ├── classification_report.csv # Precision, recall, F1 per class │ ├── classwise_accuracy.csv # Per-class validation accuracy │ ├── confusion_matrix_data.csv # Raw confusion matrix values │ └── metrics_over_epochs.csv # Train loss, valid loss, accuracy per epoch ├── project.ipynb # Clean final notebook (uses relative paths) ├── project_colab.ipynb # Training notebook for Colab (uses Drive paths) ├── README.md # Project documentation (this file) ├── requirements.txt # Dependencies list └── report.pdf # Final report (IEEE TPAMI format)

Group09_Project/
  dataset/
    train/                         # Training images (pre-split)
    valid/                         # Validation images (pre-split)

  models/
    best_alexnet.pkl              # Exported best-performing AlexNet model

  visualizations/
    confusion_matrix.png                   # Raw confusion matrix (unnormalized)
    confusion_matrix_normalized.png        # Normalized confusion matrix
    lr_find_plot.png                       # Learning rate finder curve
    lr_find_plot_labeled.png               # LR curve with valley marked
    sample_batch.png                       # Sample training batch shown before training
    sample_predictions.png                 # Grid of sample predictions (correct/incorrect)
    top_losses.png                         # Top high-loss predictions
    top_losses_examples.png                # Broader view of most confused predictions
    top_losses_per_class.png               # Misclassifications per vehicle class
    train_loss.png                         # Plot of training loss only
    train_vs_val_loss.png                  # Training vs validation loss curves
    val_accuracy_over_epochs.png           # Validation accuracy over training epochs

    classification_report.csv              # Precision, recall, F1-score per class
    classwise_accuracy.csv                 # Validation accuracy per class
    confusion_matrix_data.csv              # Numerical confusion matrix values
    metrics_over_epochs.csv                # Epoch-wise training/validation loss and accuracy

  project.ipynb                  # Clean version for final evaluation, uses relative paths
  project_colab.ipynb            # Google Colab training notebook, uses Drive paths
  README.md                      # Project documentation (this file)
  requirements.txt               # Package dependencies
  report.pdf                     # Final 8-page project report (IEEE TPAMI format)

## 🧠 Notebooks

- `project.ipynb`: Final version for grading and code review. Uses relative paths so it works in both local and Colab environments (if files are uploaded manually).
- `project_colab.ipynb`: Primary notebook used for training with Google Drive. Requires the dataset to be located at `drive/MyDrive/VehicleModelProject/dataset/`.

➡️ If using Google Colab, mount your drive and place the dataset inside: "/content/drive/MyDrive/VehicleModelProject/dataset/"


## 📊 Evaluation Artifacts

All training diagnostics, loss curves, confusion matrices, and prediction analyses are saved in the `visualizations/` folder. These are referenced throughout the report and provide insight into the model’s learning behavior, error patterns, and classification performance.

## 📄 Report

- `report.pdf`: Full technical report formatted in IEEE TPAMI style. Includes background, methodology, results, analysis, ablation studies, limitations, and future directions.
