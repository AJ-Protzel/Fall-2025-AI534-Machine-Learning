rt warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*Ill-conditioned matrix \(rcond=.*\): result may not be accurate\.",
            module=r".*scipy\._lib\._util.*"
        )