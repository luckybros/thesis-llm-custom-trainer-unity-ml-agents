from setuptools import setup, find_packages

try:
    from mlagents.plugins import ML_AGENTS_TRAINER_TYPE
    ENTRY_POINTS = {
        ML_AGENTS_TRAINER_TYPE: [
            "llm_trainer=mlagents_plugin.trainers.trainer.llm_trainer:get_type_and_setting",
        ]
    }
except ImportError:
    ENTRY_POINTS = {}

setup(
    name="mlagents_trainer_plugin",
    version="0.0.1",
    packages=find_packages(),
    entry_points=ENTRY_POINTS,
    install_requires=[],
    extras_requires={
        "trainer": ["mlagents"],
        "llm_server": ["pyzmq"]
    }
)