from isofit.inversion.inverse import Inversion as ClassicInversion


def Inversion(config, fm):
    """
    Retrieves the correct Inversion model to initialize and returns
    """

    if config.implementation.mode in ("inversion", "simulation"):
        return ClassicInversion(config, fm)

    else:
        # This should never be reached due to configuration checking
        raise AttributeError("Config implementation mode node valid")
