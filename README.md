# landmark-seed-finder

## Pokemon: Legends Arceus landmark (tree/rock) seed finder from a PA8 dumps of pokemon

- Usage:
	- Place .pa8 files next to executable (.exe or .py) with the name format ``[Order Caught]-[Map Index]-[Landmark ID].pa8``
		- E.g.: 0-0-0807.pa8 would be first Pokémon caught (0), on map Obsidian Fieldlands (0) with map ID 0807.

		- Map indexes:
			- 0: Obsidian Fieldlands
			- 1: Crimson Mirelands
			- 2: Cobalt Coastlands
			- 3: Coronet Highlands
			- 4: Alabaster Icelands

		- Landmark ID: https://lincoln-lm.github.io/JS-Finder/Gen8/PLA-Landmark-Map/

	- Edit config.toml (text file) to specify the settings to search for each landmark
		- ``shiny_only``
			- Whether or not to filter for shinies (true, false)
		- ``alpha_only``
			- Whether or not to filter for alphas (true, false)
		- ``max_advances``
			- Maximum number of advances to check (integer)
		- ``shiny_rolls``
			- Number of shiny rolls the pokemon generate with (integer)
		- ``max_gap``
			- Maximum gap between landmark & fixed seeds to check (leave at 4 unless you know what youre doing) (integer)
