# landmark-seed-finder

Changes compared to the original version:

-Iterates over each .pa8 file in the folder. .pa8 files need to have a certain format: [order caught]-[map index]-[map ID],
	e.g.: 0-0-0807.pa8 would be first Pokémon caught (0), on map Obsidian fieldlands (0) with map ID 0807.

	The list of maps with their respective IDs are as follows:
		0: Obsidian fieldlands
		1: Crimson mirelands
		2: Cobalt coastlands
		3: Coronet highlands
		4: Alabaster icelands

	For the map ID of the landmark, please refer to this link:
		https://lincoln-lm.github.io/JS-Finder/Gen8/PLA-Landmark-Map/

-Added a config.txt for the user to select their own preferences:
	-Searching specifically for only shiny, only alpha or both
	-Selecting minimum or maximum advances

-Changed the output to display gender and natures in full letters