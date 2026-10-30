"""Starter scenes for the App: famous shot / reverse-shot dialog, where the back-and-forth across cuts is the joke.
Clips are fetched whole at runtime (never committed), at the lowest resolution YouTube offers; the App's Trim Video
picks the part to swap (the first 60 s by default).
Swaps describe what is VISIBLE ("the bald man with glasses"), not roles ("Walter"): that is what d1 can see."""

STARTERS = {
    "The Social Network: the breakup": {
        "url": "https://www.youtube.com/watch?v=VlSkPA60ujQ",
        "swaps": "the young man in the gray hoodie = cat"},
    "Pulp Fiction: Royale with Cheese": {
        "url": "https://www.youtube.com/watch?v=6Pkq_eBHXJ4",
        "swaps": "the man with long dark hair in a ponytail = dog\nthe man with curly hair and sideburns = goose"},
    "Breaking Bad: I am the one who knocks": {
        "url": "https://www.youtube.com/watch?v=Ca3kPemW2CE",
        "swaps": "the bald man with a goatee = cat"},
    "Succession: Boar on the Floor": {
        "url": "https://www.youtube.com/watch?v=dYqqW3c2mBU",
        "swaps": "the old man with white hair = camel"},
    "Mean Girls: You can't sit with us": {
        "url": "https://www.youtube.com/watch?v=_bXSVc8lN7o",
        "swaps": "the blonde girl in the pink jacket = goose"},
    "Casablanca: As Time Goes By": {
        "url": "https://www.youtube.com/watch?v=a_paHK6JdHA",
        "swaps": "the woman in the white dress = cat\nthe man in the white jacket and bow tie = parrot"},
}
MY_VIDEO = "my video (upload)"

ANIMAL_IDEAS = ["cat", "dog", "parrot", "goose", "camel", "llama", "capybara", "penguin", "goat", "raccoon",
                "corgi", "owl", "alpaca", "pig", "rabbit", "duck", "hedgehog", "a fluffy white goat"]


def parse_swaps(text):
    """'who = animal' per line -> [(who, animal)]. Blank lines and lines without '=' are ignored."""
    out = []
    for line in text.splitlines():
        if "=" in line:
            who, animal = line.rsplit("=", 1)
            if who.strip() and animal.strip():
                out.append((who.strip(), animal.strip()))
    return out
