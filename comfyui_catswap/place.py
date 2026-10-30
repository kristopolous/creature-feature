"""Where the animal goes and what it is told to do. Pure geometry and text, no models.
Any animal works (FLUX 3 Klein draws whatever is named); the tables only make sizes and manners better.

Boxes here are FLUX 3 grid boxes: [y_min, x_min, y_max, x_max] on 0..1000.
"""
import json

# Height of the animal in its natural pose, as a fraction of a standing adult's height.
HEIGHT = {
    "mouse": 0.03, "hamster": 0.03, "frog": 0.03, "turtle": 0.08, "hedgehog": 0.06, "guinea pig": 0.06,
    "rabbit": 0.15, "bunny": 0.15, "owl": 0.15, "parrot": 0.15, "alligator": 0.15, "crocodile": 0.15,
    "chicken": 0.2, "duck": 0.2, "raccoon": 0.2, "corgi": 0.2, "cat": 0.22, "kitten": 0.12, "fox": 0.25,
    "monkey": 0.3, "sloth": 0.3, "capybara": 0.3, "dog": 0.35, "penguin": 0.35, "goose": 0.4,
    "pig": 0.45, "chimpanzee": 0.5, "sheep": 0.5, "goat": 0.5, "panda": 0.55, "tiger": 0.6, "lion": 0.65,
    "deer": 0.7, "flamingo": 0.7, "donkey": 0.75, "bear": 0.8, "kangaroo": 0.8, "cow": 0.8, "gorilla": 0.9,
    "horse": 1.0, "llama": 1.0, "alpaca": 0.9, "camel": 1.15, "ostrich": 1.3, "moose": 1.2, "elephant": 1.6,
    "giraffe": 2.9,
}
# Width / height of the animal's silhouette in its natural pose (default 1.0).
ASPECT = {"cat": 0.8, "rabbit": 0.9, "penguin": 0.5, "owl": 0.6, "parrot": 0.5, "flamingo": 0.6,
          "ostrich": 0.6, "giraffe": 0.5, "kangaroo": 0.6, "meerkat": 0.4, "horse": 1.3, "cow": 1.4,
          "camel": 1.2, "pig": 1.4, "sheep": 1.3, "goat": 1.2, "dog": 1.1, "corgi": 1.5, "crocodile": 4.0,
          "alligator": 4.0, "elephant": 1.3, "tiger": 1.6, "lion": 1.5, "bear": 1.4, "llama": 0.9}
DEFAULT_HEIGHT = 0.35

# A few animals with signature manners. Everyone else gets the generic ones below.
MANNERS = {
    "cat": "licks its paw, then stares at nothing",
    "dog": "pants, scratches its ear and sniffs the air",
    "monkey": "scratches its head and picks at its fur",
    "rabbit": "twitches its nose and nibbles at nothing",
    "camel": "chews slowly with a bored look",
    "goose": "flaps its wings once and honks at no one",
    "goat": "chews and stares blankly",
    "penguin": "waddles a step and shuffles its flippers",
    "horse": "flicks its ears and shifts its weight",
}


def key(animal):
    """Lookup key: the last word that names a known animal ("a fluffy white rabbit" -> "rabbit")."""
    a = animal.lower().strip()
    for k in sorted(HEIGHT, key=len, reverse=True):
        if k in a:
            return k
    return a.split()[-1] if a else "cat"


def article(animal):
    a = animal.strip()
    return a if a.lower().startswith(("a ", "an ", "the ")) else ("an " if a[:1].lower() in "aeiou" else "a ") + a


def animal_box(person, posture, frame_aspect, touches_bottom, animal="cat", size=0.0):
    """Where a real animal of its real size would be, given the person's grid box.
    size: animal height as a fraction of a standing adult; 0 = look it up."""
    ratio = size or HEIGHT.get(key(animal), DEFAULT_HEIGHT)
    aspect = ASPECT.get(key(animal), 1.0)
    y0, x0, y1, x1 = person
    ph, pw = y1 - y0, x1 - x0
    cx = (x0 + x1) / 2
    if touches_bottom:
        # medium/close shot, feet out of frame: the animal takes the person's place in frame (on the seat,
        # at the table), rising from the bottom edge; at least half their visible height so it reads
        adult = ph / 0.55
        top, bottom = 1000 - max(ratio * adult, 0.5 * ph), 1000
    elif posture == "sitting":  # on the seat; a seated person's box is ~0.72 of their standing height
        adult = ph / 0.72
        bottom = y0 + 0.62 * ph
        top = bottom - max(ratio * adult, 0.2 * ph)
    else:  # standing / walking / lying: on the ground at their feet
        adult = ph if posture != "lying" else pw
        top, bottom = y1 - ratio * adult, y1
    top = max(0, top)
    half = (bottom - top) * aspect / frame_aspect / 2  # grid x/y units differ by the frame aspect
    half = max(half, 10)
    return [round(top), round(max(0, cx - half)), round(bottom), round(min(1000, cx + half))]


def pose_words(posture, touches_bottom):
    if touches_bottom:
        return "sitting where the person was"
    return {"standing": "sitting on the ground", "walking": "standing on its feet",
            "sitting": "sitting on the seat", "lying": "lying down"}.get(posture, "sitting")


def swap_record(person, animal_bbox, animal="cat", posture="standing", mode="replace", touches_bottom=False):
    """FLUX 3 / Klein i2i_bbox record that removes the person and adds the animal.
    mode="replace": one from-null row. mode="two_rows": remove row (tgt null) + add row (from null)."""
    desc = f"{article(animal)}, an ordinary real {key(animal)}"
    para = (f"Replace the person <person_1> with {article(animal)} <animal_1>. It is a real animal at its natural "
            f"size, in a natural animal pose, not anthropomorphic, no clothes. Fill in the background where the "
            f"person was. Match the scene's lighting, color grade, film grain and perspective. Everything else "
            f"stays identical to the source.")
    add = {"id": "animal_1", "from": None, "src_bbox": None, "tgt_bbox": list(animal_bbox),
           "desc": f"{desc}, {pose_words(posture, touches_bottom)}, looking off to the side"}
    if mode == "replace":
        rows = [{**add, "src_bbox": list(person)}]
    else:
        rows = [{"id": "person_1", "from": "ref_image_0", "src_bbox": list(person), "tgt_bbox": None,
                 "desc": "the person, removed"}, add]
    return f"{para} {json.dumps(rows)}"


def motion_prompt(animal="cat", posture="standing", scene="", touches_bottom=False):
    """LTX prompt for the regenerated region. The rest of the frame is frozen original footage."""
    k = key(animal)
    manner = MANNERS.get(k, "looks around, sniffs the air and fidgets")
    where = "where the person was" if touches_bottom or posture == "sitting" else "on the ground"
    move = "walks along slowly, then stops" if posture == "walking" else f"stays {where}"
    scene = f"{scene.rstrip('.')}. " if scene else ""
    return (f"{scene}{article(animal).capitalize()} {move}, {manner}. It behaves exactly like a normal {k} and pays "
            f"no attention to the conversation around it. Realistic fur and anatomy, natural animal movement, same "
            f"lighting and film look as the rest of the shot. Static camera.")


# Two-pass swap (what the App uses). One pass can't erase a person who fills a third of the frame: Klein adds the
# animal and keeps the person. Removing first gives a clean plate; the animal is then added into it.
def remove_record(person, who="the person"):
    para = (f"Remove {who} <person_1> completely. Fill in the background behind them: the room, walls, furniture "
            f"and lights continue naturally. No person remains. Match the lighting, color grade and film grain; "
            f"everything else stays identical.")
    rows = [{"id": "person_1", "from": "ref_image_0", "src_bbox": list(person), "tgt_bbox": None, "desc": who}]
    return f"{para} {json.dumps(rows)}"


def add_record(animal_bbox, animal="cat", posture="standing", touches_bottom=False):
    para = (f"Add {article(animal)} <animal_1> to the scene. It is a real animal at its natural size, in a natural "
            f"animal pose, not anthropomorphic, no clothes. Match the lighting, color grade, film grain and "
            f"perspective; everything else stays identical.")
    rows = [{"id": "animal_1", "from": None, "src_bbox": None, "tgt_bbox": list(animal_bbox),
             "desc": f"{article(animal)}, an ordinary real {key(animal)}, {pose_words(posture, touches_bottom)}, "
                     f"looking off to the side"}]
    return f"{para} {json.dumps(rows)}"
