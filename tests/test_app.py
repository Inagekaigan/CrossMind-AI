from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_workbench_opens():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app/main.py'), default_timeout=90)
    app.run()
    assert not app.exception
    assert app.title[0].value == 'Observe motion. Investigate what comes next.'


def test_trained_model_and_recorded_demo():
    root = Path(__file__).resolve().parents[1]
    if not (root/'experiments/jaad_default/joint_17/best.pt').exists():
        import pytest
        pytest.skip('Integration test requires locally trained checkpoint')
    app = AppTest.from_file(str(root/'app/main.py'), default_timeout=90).run()
    app.selectbox[0].select('joint_17').run()
    assert not app.exception
    assert app.metric[0].value != 'Unavailable'
    demo_button = next(b for b in app.button if b.label == 'Open recorded JAAD demonstration')
    demo_button.click().run()
    assert not app.exception
    assert any(m.label == 'Tracks observed' and m.value == '3' for m in app.metric)
