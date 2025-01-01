import torch
import timm
import torch.onnx


def convert_vit_to_onnx(model_checkpoint_path, onnx_output_path, num_classes=47):
    """
    Convert a Vision Transformer (ViT) model to ONNX format using torch.onnx.dynamo_export.

    Args:
        model_checkpoint_path (str): Path to the model checkpoint
        onnx_output_path (str): Path to save the ONNX model
        num_classes (int): Number of output classes
    """
    # Determine the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = torch.load(model_checkpoint_path, map_location=device)

    # Set the model to evaluation mode
    model.eval()

    # Create a dummy input
    x = torch.randn(1, 3, 224, 224, requires_grad=False).to(device)

    try:
        exported_model = torch.onnx.dynamo_export(model,
                                                  x,
                                                  export_options=torch.onnx.ExportOptions(dynamic_shapes=True)
                                                  )

        # Save the exported model
        exported_model.save(onnx_output_path)

        print(f"Model successfully exported to {onnx_output_path}")

    except Exception as e:
        print(f"Error during ONNX export: {e}")
        import traceback
        traceback.print_exc()



def verify_onnx_model(onnx_path):
    """
    Verify the exported ONNX model
    """
    import onnx

    try:
        # Load the ONNX model
        onnx_model = onnx.load(onnx_path)

        # Check the model
        onnx.checker.check_model(onnx_model)

        print("ONNX model is valid!")

        # Optional: Print model information
        print("\nModel Graph:")
        print(onnx.helper.printable_graph(onnx_model.graph))

    except Exception as e:
        print(f"Error verifying ONNX model: {e}")


if __name__ == '__main__':
    convert_vit_to_onnx(
        model_checkpoint_path='../../models/plant-classifier-vitb32.pth',
        onnx_output_path='../../models/plant-classifier-vitb32.onnx',
        num_classes=47
    )

    verify_onnx_model('../../models/plant-classifier-vitb32.onnx')