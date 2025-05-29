
from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as go
def create_bbox_trace(corners, color, name):
    """Create plotly trace for 3D bounding box"""
    # Define the 12 edges of a cuboid
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]

    x_coords = []
    y_coords = []
    z_coords = []

    for edge in edges:
        x_coords.extend([corners[edge[0], 0], corners[edge[1], 0], None])
        y_coords.extend([corners[edge[0], 1], corners[edge[1], 1], None])
        z_coords.extend([corners[edge[0], 2], corners[edge[1], 2], None])

    return go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='lines',
        line=dict(color=color, width=4),
        name=name
    )

def visualize_3d_predictions(rgb_image, gt_corners, pred_corners, gt_conf, pred_conf,
                           pointcloud=None, save_path=None):
    """
    Create comprehensive visualization of 3D bbox predictions
    """
    num_objects = len(gt_conf)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('RGB Image', '3D Scene with Predictions',
                       'Ground Truth vs Predicted', 'Confidence Scores'),
        specs=[[{"type": "xy"}, {"type": "scene"}],
               [{"type": "scene"}, {"type": "bar"}]]
    )

    # 1. RGB Image
    if rgb_image is not None:
        fig.add_trace(
            go.Image(z=rgb_image),
            row=1, col=1
        )

    # 2. 3D Scene with Point Cloud and Bboxes
    if pointcloud is not None:
        # Subsample point cloud for visualization
        if len(pointcloud) > 5000:
            indices = np.random.choice(len(pointcloud), 5000, replace=False)
            pc_vis = pointcloud[indices]
        else:
            pc_vis = pointcloud

        fig.add_trace(
            go.Scatter3d(
                x=pc_vis[:, 0], y=pc_vis[:, 1], z=pc_vis[:, 2],
                mode='markers',
                marker=dict(size=1, color='lightgray', opacity=0.3),
                name='Point Cloud'
            ),
            row=1, col=2
        )

    # Add ground truth and predicted bboxes
    colors_gt = ['green', 'blue', 'red', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    colors_pred = ['lightgreen', 'lightblue', 'pink', 'yellow', 'lavender', 'aqua', 'fuchsia', 'lightyellow']

    for i, (gt_box, pred_box) in enumerate(zip(gt_corners, pred_corners)):
        if not (gt_box == 0).all():  # Valid ground truth
            # Ground truth bbox
            fig.add_trace(
                create_bbox_trace(gt_box, colors_gt[i % len(colors_gt)], f'GT Box {i}'),
                row=1, col=2
            )

            # Predicted bbox
            fig.add_trace(
                create_bbox_trace(pred_box, colors_pred[i % len(colors_pred)], f'Pred Box {i}'),
                row=1, col=2
            )

    # 3. Side-by-side comparison
    for i, (gt_box, pred_box) in enumerate(zip(gt_corners, pred_corners)):
        if not (gt_box == 0).all():
            fig.add_trace(
                create_bbox_trace(gt_box, colors_gt[i % len(colors_gt)], f'GT {i}'),
                row=2, col=1
            )
            fig.add_trace(
                create_bbox_trace(pred_box, colors_pred[i % len(colors_pred)], f'Pred {i}'),
                row=2, col=1
            )

    # 4. Confidence comparison
    object_names = [f'Object {i}' for i in range(num_objects)]
    fig.add_trace(
        go.Bar(
            x=object_names,
            y=gt_conf,
            name='GT Confidence',
            marker_color='blue'
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(
            x=object_names,
            y=pred_conf,
            name='Pred Confidence',
            marker_color='red'
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title_text="3D Bounding Box Prediction Results",
        showlegend=True,
        height=800,
        width=1200
    )

    if save_path:
        fig.write_html(save_path)


def plot_training_metrics(train_losses, val_losses, save_path=None):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Total loss
    axes[0, 0].plot(train_losses['total'], label='Train', color='blue')
    axes[0, 0].plot(val_losses['total'], label='Validation', color='red')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Bbox loss
    axes[0, 1].plot(train_losses['bbox'], label='Train', color='blue')
    axes[0, 1].plot(val_losses['bbox'], label='Validation', color='red')
    axes[0, 1].set_title('Bounding Box Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Confidence loss
    axes[1, 0].plot(train_losses['conf'], label='Train', color='blue')
    axes[1, 0].plot(val_losses['conf'], label='Validation', color='red')
    axes[1, 0].set_title('Confidence Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Learning rate (if available)
    axes[1, 1].text(0.5, 0.5, 'Additional metrics\ncan be added here',
                   transform=axes[1, 1].transAxes, ha='center', va='center')
    axes[1, 1].set_title('Additional Metrics')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

