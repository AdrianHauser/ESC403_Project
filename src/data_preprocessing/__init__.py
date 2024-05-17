class Pipeline:
    """Base class for pipelines."""

    def __init__(self, download: bool):
        self.download = download

    def _setup(self):
        """Setup phase."""
        pass

    def _run(self, *args, **kwargs):
        """Run phase."""
        pass

    def _cleanup(self):
        """Cleanup after import."""
        pass

    def _teardown(self):
        """Teardown after import."""
        pass

    def run(self, *args, **kwargs):
        setup_result = self._setup()
        run_result = self._run(**setup_result)
