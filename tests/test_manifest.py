import pytest
from scripts.prepare_dataset import read_manifest


def test_subject_leakage_rejected(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text("path,exercise,subject_id,split\na.mp4,squat,person1,train\nb.mp4,squat,person1,test\n")
    with pytest.raises(ValueError, match="multiple splits"):
        read_manifest(path, "test")
