import os
import glob
import click
import shutil


@click.command()
@click.argument(
    "target",
    type=click.Path(exists=True, file_okay=False, writable=True),
    default=os.getcwd(),
)
def copy_notebooks(target):
    """Copy example notebooks into the TARGET directory"""

    # Locate notebook files in the installation tree
    jupyter_dir = os.path.join(os.path.split(__file__)[0], "notebooks")
    print(jupyter_dir)
    notebooks = glob.glob(os.path.join(jupyter_dir, "*.ipynb"))

    # Print a verbose message
    click.echo(f"Copying {len(notebooks)} Jupyter notebooks into directory '{target}'")

    # Do the actual copying
    for notebook in notebooks:
        shutil.copy(notebook, target)
