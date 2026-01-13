from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class NullImageRunner(BaseImageRunner):
    """No-op runner for offline-only training."""

    def run(self, policy):
        return {}
