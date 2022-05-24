from ae_anom import AEAnom
from ae_basic import AEBasic
from ae_gen import AEGen
from vae_anom import VAEAnom
from vae_basic import VAEBasic
from vae_gen import VAEGen
"""
Main function to run different tasks from
"""

if __name__ == "__main__":
    task = AEBasic(three_colors=False)  # AE-basic (mono and stacked)
    # task = AEGen(three_colors=False)  # AE-gen (mono and stacked)
    # task = AEAnom(three_colors=False)  # AE-anom (mono and stacked)
    # task = VAEBasic(three_colors=False)  # VAE-basic (mono and stacked)
    # task = VAEGen(three_colors=False)  # VAE-gen (mono and stacked)
    # task = VAEAnom(three_colors=False)  # VAE-anom (mono and stacked)

    task.run()
