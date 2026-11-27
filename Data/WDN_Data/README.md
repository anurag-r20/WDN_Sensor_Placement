# Water distribution network data

Every network here except `Test_WDN` comes from the [Water Distribution System Research Database](https://uknowledge.uky.edu/wdsrd/) at the University of Kentucky (DOI [10.13023/kwrri.wdsrd](https://doi.org/10.13023/kwrri.wdsrd)). `Test_WDN` is a small hand-made example.

## Layout

Each `<Name>_WDN` folder holds the network's EPANET file and three comma-separated files that the notebooks read.

- `<Name>_WDN_Nodes.txt` lists each node with its X and Y coordinates.
- `<Name>_WDN_Edges.txt` lists each edge by its two end nodes.
- `<Name>_WDN_Water_Consumption.txt` gives each node's consumption.

`Source.txt` holds the database's recommended citation for the network and its license. Some folders also keep the system description PDF and the KYPIPE `.p2k` file.

## Conversion

Apulia, Fossolo, Kentucky (KY 1), Modena, Pescara and ZJ were converted by hand. The other networks were converted from their EPANET files with one rule, which keeps the whole physical network. Every junction, reservoir and tank is a node, and every pipe, pump and valve is an edge. Node IDs that are bare numbers get a `J-` (junction) or `T-` (reservoir or tank) prefix, as in the Modena files. A junction's consumption is its base demand, or the sum of its `[DEMANDS]` entries when it has any, and reservoirs and tanks have none. A few nodes have no coordinates in their EPANET file, so their X and Y are empty.

Seven of the database's EPANET files are in EPANET's binary project format rather than text. These are Exnet, Two-Loop, Modified FOWM and KYPIPE Examples 1, 2, 5 and 8. Their folders hold only the original file until it is exported from the EPANET app as a text `.inp` and converted.

## License

The database releases every network under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Reuse must credit the network's authors as given in its `Source.txt`, and it is limited to non-commercial purposes. The TXT files are adaptations of the EPANET files.
