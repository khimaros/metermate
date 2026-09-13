"""a notification never arrives without something on the verdict page (r4.5).

the crop a phone attaches is the file the verdict page lists, so the two are one
fact seen twice. they used to be decided separately: a track stays confirmed for
as long as it is in view and the sighting arm notified on every look of it, while
`Alerting::confirmed` hands back crop names only the first time -- so the first
notification had a picture and the rest had nothing, and a vehicle that confirmed
before the harvest kept one of its place could notify with no page entry at all.

these drive that path for real: a clip with a vehicle in it, references trained
from the same street, and a fake ntfy server recording what the phone would have
received.
"""

import re
from pathlib import Path

import pytest

from alerts import SUBJECT, FakeNtfy, alerting_config, run, trained

CONFIRMED = re.compile(rf"{SUBJECT.upper()} at ")
# what `[ntfy] click` holds once the phone can reach the preview.
PREVIEW = "http://preview.example.com:8420"


@pytest.fixture(scope="module")
def street(binary, tmp_path_factory) -> Path:
    """the work directory: the clips, and references trained from their own crops.

    the expensive part, and nothing about it depends on whether the harvest is
    on, which is the only thing the two tests below differ in.
    """
    return trained(binary, tmp_path_factory.mktemp("evidence"))


def test_every_notification_carries_a_crop_the_harvest_holds(binary, street, tmp_path):
    """**the attachment is the invariant.**

    a notification that goes without one is a claim about a vehicle nothing
    recorded, and the only proof available is that the file is on disk under the
    subject it announced.
    """
    crops = tmp_path / "crops"
    with FakeNtfy() as ntfy:
        cfg = alerting_config(street / "big", ntfy.url, crops, street / "trained", click=PREVIEW)
        code, out = run(binary, cfg)

    assert code == 0, out
    assert CONFIRMED.search(out), f"nothing ever confirmed:\n{out}"
    assert ntfy.seen, f"a confirmed subject notified nobody:\n{out}"

    for note in ntfy.seen:
        # the words travel in the headers only when the body is the picture, so
        # a POST here is a notification that went out with nothing attached.
        assert note["method"] == "PUT", f"no evidence attached: {note['headers']}"
        name = Path(note["headers"]["filename"]).name
        saved = crops / name
        assert saved.exists(), f"{name} was notified and never written:\n{out}"
        assert f"_{SUBJECT}_" in name, f"{name} does not name what it announced"
        assert note["body"] == saved.read_bytes()
        # **and the tap arrives at it.** the crop is the reason the notification
        # was sent, so it is what a tap should open -- the same file the verdict
        # page lists, which is the guarantee r4.5 is built on.
        assert note["headers"]["click"] == f"{PREVIEW}/#/verdict/{name}", note["headers"]


def test_a_confirmation_with_nothing_to_show_is_logged_and_held_back(binary, street, tmp_path):
    """**no page entry, no phone.** with the harvest off the confirming look has
    nowhere to write, so the alert has nothing behind it and says so rather than
    buzzing: the failure the requirement is about, made to happen on purpose."""
    with FakeNtfy() as ntfy:
        cfg = alerting_config(
            street / "big", ntfy.url, tmp_path / "crops", street / "trained", harvest=False
        )
        code, out = run(binary, cfg)

    assert code == 0, out
    assert not ntfy.seen, f"a subject with no crop notified anyway: {ntfy.seen}"
    assert "not alerting" in out, out
