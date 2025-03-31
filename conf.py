SOURCES = [
    {"position": (-4.0, 3.0), "filename": "wav/5-181977-A-35.wav"},
    # {"position": (0.75, 6.0), "filename": "wav/392.wav"},
    # {"position": (3.5, 2.0), "filename": "wav/440.wav"},
]

c = 343
# print(c / 440)
optimal_spacing = c / (2 * 4e3)
# optimal_spacing = 0.5

# print(f"{optimal_spacing=}")


n_mics = 32
total_span = (n_mics - 1) * optimal_spacing
# start_x = -total_span / 2
start_x = 0

MICROPHONES = []
for i in range(n_mics):
    m = {}
    m["filename"] = f"micinput/mic{i + 1:02d}.wav"
    m["position"] = (start_x + i * optimal_spacing, 0)
    MICROPHONES.append(m)
