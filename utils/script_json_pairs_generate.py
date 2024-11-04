import json
import os


def generate_json(mesh_filenames, output_filename):
    pairs = []

    for mesh in mesh_filenames:
        # Extract base name without extension
        base_name = os.path.splitext(os.path.basename(mesh))[0]

        # Create centers filename based on the given pattern
        centers_filename = f"{base_name}res_1000_{base_name[-3:]}.xyz"  # Assumes last 3 characters are the index
        centers_filepath = os.path.join(os.path.dirname(mesh), centers_filename)

        # Append to pairs list
        pairs.append({
            "mesh_filename": mesh,
            "centers_filename": centers_filepath
        })

    # Create the output dictionary
    output_data = {"pairs": pairs}

    # Save to JSON file
    with open(output_filename, 'w') as json_file:
        json.dump(output_data, json_file, indent=2)


# Example usage
if __name__ == "__main__":
    # List of mesh filenames
    mesh_filenames = [
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball000.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball001.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball002.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball003.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball004.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball005.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball006.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball007.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball008.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball009.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball010.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball011.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball012.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball013.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball014.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball015.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball016.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball017.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball018.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball019.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball020.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball021.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball022.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball023.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball024.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball025.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball026.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball027.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball028.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball029.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball030.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball031.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball032.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball033.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball034.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball035.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball036.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball037.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball038.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball039.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball040.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball041.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball042.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball043.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball044.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball045.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball046.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball047.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball048.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball049.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball050.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball051.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball052.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball053.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball054.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball055.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball056.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball057.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball058.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ball059.obj"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_000.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_001.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_002.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_003.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_004.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_005.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_006.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_007.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_008.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_009.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_010.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_011.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_012.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_013.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_014.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_015.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_016.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_017.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_018.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_019.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_020.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_021.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_022.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_023.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_024.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_025.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_026.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_027.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_028.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_029.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_030.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_031.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_032.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_033.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_034.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_035.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_036.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_037.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_038.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_039.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_040.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_041.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_042.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_043.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_044.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_045.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_046.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_047.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_048.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_049.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_050.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_051.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_052.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_053.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_054.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_055.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_056.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_057.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_058.xyz"
    "G:\Můj disk\00_MAIN\0_škola\FAV\4.rocnik_24_25\Bakalarka\_projekt\FAV_BP_24_25_Parametrization\data\raw\ball\ballres_1000_059.xyz"
    ]


    # Output JSON filename
    output_filename = "output.json"

    # Generate JSON
    generate_json(mesh_filenames, output_filename)

    print(f"JSON saved to {output_filename}")
