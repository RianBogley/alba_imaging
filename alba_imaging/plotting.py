# %%
# ---------------------------------------------------------------------------- #
#                               IMPORT LIBRARIES                               #
# ---------------------------------------------------------------------------- #
import nilearn.plotting as plotting
import matplotlib.pyplot as plt

# %%
# ---------------------------------------------------------------------------- #
#                               DEFINE FUNCTIONS                               #
# ---------------------------------------------------------------------------- #

# Plot nifti image over template in html view
def html_brain_plot(nifti,template):
    """
    """
    html_brain = plotting.view_img(nifti,
                                colorbar=True,
                                threshold=0,
                                black_bg=False,
                                plot_abs=False,
                                display_mode='lyrz',
                                bg_img=template,
                                cmap='bwr',
                                dpi=300,
                                resampling_interpolation='nearest',)
    return html_brain

def view_img_plot(nifti,template):
    """
    """
    view_img = plotting.view_img(nifti,
                                colorbar=True,
                                threshold=0,
                                black_bg=False,
                                plot_abs=False,
                                display_mode='lyrz',
                                bg_img=template,
                                cmap='bwr',
                                dpi=300,
                                resampling_interpolation='nearest',)
    return view_img

# Plot nifti image as glass brain figure
def glass_brain_plot(nifti):
    """
    """
    glass_brain = plotting.plot_glass_brain(nifti,
                                            threshold=0,
                                            colorbar=True,
                                            black_bg=False,
                                            plot_abs=False,
                                            display_mode='lyrz',
                                            cmap='bwr')
    return glass_brain




# Plot the mean absolute error for a prediction model
def mean_abs_err_plot(y_test_final,y_pred,prediction_score,main_dir,run_name):
    """
    Plot the mean absolute error for the prediction.
    """
    plt.figure(figsize=(6, 4.5))
    plt.suptitle(f"Mean Absolute Error = {prediction_score}", fontsize=16)
    linewidth = 3
    plt.plot(y_test_final, label="True values", linewidth=linewidth)
    plt.plot(y_pred,"--",c="g", label="Predicted values", linewidth=linewidth)
    plt.ylabel("Score")
    plt.xlabel("Subject")
    plt.legend(loc="best")
    plt.savefig(f'{main_dir}/{run_name}_mae.png', dpi=300, bbox_inches='tight')

# Plot the true minus predicted values for the prediction model
def true_minus_pred_plot(y_test_final,y_pred,main_dir,run_name):
    """
    Plot the true minus predicted values.
    """
    plt.figure(figsize=(6, 4.5))
    plt.suptitle(f"True - Predicted Values", fontsize=16)
    linewidth = 3
    plt.plot(y_test_final - y_pred, label="True - Predicted", linewidth=linewidth)
    plt.ylabel("Score")
    plt.xlabel("Subject")
    plt.legend(loc="best")
    plt.savefig(f'{main_dir}/{run_name}_true_minus_pred.png', dpi=300, bbox_inches='tight')
