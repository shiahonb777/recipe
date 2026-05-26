"""recipe — ship model recipes, not weights.

A recipe is a small text+adapter bundle that fully determines a fine-tuned
model. Instead of uploading a 14 GB merged checkpoint to HF Hub, you push
a 50 KB recipe + (optionally) the LoRA adapter. Anyone with the base model
referenced in the recipe can rebuild the merged weights bit-exactly on
their own machine.

Public API:
    Recipe         — the recipe data structure
    load_recipe    — read a recipe from disk
    save_recipe    — write a recipe to disk
    materialize    — apply the recipe (download base + adapter + merge)
"""

from __future__ import annotations

from mlrecipe.recipe import Recipe, load_recipe, save_recipe
from mlrecipe.materialize import materialize

__version__ = "0.1.0"
__all__ = ["Recipe", "load_recipe", "save_recipe", "materialize", "__version__"]
