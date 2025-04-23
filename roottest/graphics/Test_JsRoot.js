//IMPORTS
const rootSys = process.env.ROOTSYS;
const path_jsroot = `${rootSys}/js/modules/main.mjs`;
const { version, parse, makeSVG } = await import(path_jsroot);

import { promises as fs } from 'fs';
import path from 'path';
import xmlParser from 'xml-parser-xo';
import chalk from 'chalk';

// JSROOT version
console.log(chalk.blue(`JSROOT version ${version}`));

//FUNCTIONS
async function compareSVG(svgPro, svgRef, baseName, svgRefPath) {
    try {
        const parsedProSVG = xmlParser(svgPro);
        const parsedRefSVG = xmlParser(svgRef);

        if (JSON.stringify(parsedProSVG) === JSON.stringify(parsedRefSVG)) {
            console.log(chalk.green(`MATCH: ${baseName} - Lengths [Pro: ${svgPro.length}, Ref: ${svgRef.length}]`));
            return true;
        } else {
            console.error(chalk.red(`DIFF: ${baseName} - Lengths [Pro: ${svgPro.length}, Ref: ${svgRef.length}]`));
            // Overwrite the reference SVG file with the produced one
            await fs.writeFile(svgRefPath, svgPro);
            await console.log(chalk.yellow("Reference SVG file updated"));
            throw error;
        }
    } catch (error) {
        throw error;
    }
}
//Creates an SVG from a JSON file.
async function createSVGFromJSON(filePath, builddir) {
    const baseName = path.basename(filePath, path.extname(filePath));
    const svgRefPath = `./svg_ref/${baseName}.svg`;
    const svgProPath = builddir + `/svg_pro/${baseName}_pro.svg`;

    try {
        // Read and parse JSON data
        const jsonData = await fs.readFile(filePath, 'utf8');
        const data = JSON.parse(jsonData);

        // Create SVG from parsed data
        let obj = parse(data);
        let svgPro = await makeSVG({ object: obj, option: 'lego2,pal50', width: 1200, height: 800 });

        try {
            // Check if reference SVG file exists
            await fs.access(svgRefPath);
            const svgRef = await fs.readFile(svgRefPath, 'utf8');

            // Save the produced SVG file
            await fs.writeFile(svgProPath, svgPro);

            // Compare the produced SVG with the reference SVG
            compareSVG(svgPro, svgRef, baseName, svgRefPath);
            return true;

        } catch (error) {
            // Reference file does not exist, create a new one
            if (error.code === 'ENOENT') {
                await fs.writeFile(svgRefPath, svgPro);
                console.log("Create a new reference file");
                return false;
            } else {
                console.error(chalk.red('Error accessing or reading the reference SVG file:'), error);
                return false;
            }
        }
    } catch (error) {
        console.error(chalk.red('Failed to process JSON file or create SVG:'), error);
        return false;
    }
}

/**
 * Main function to run the tests.
 */
async function main() {
    const macro = process.argv[2];
    if (!macro) {
        console.error(chalk.red('No macro specified'));
        process.exit(1);
    }
    const builddir = process.argv[3];
    if (!macro) {
        console.error(chalk.red('No builddir specified'));
        process.exit(1);
    }

    const jsonFilePath = `./json_ref/${macro}.json`;
    try {
        const success = await createSVGFromJSON(jsonFilePath, builddir);
        if (!success) {
            process.exit(1);
        }
    } catch (error) {
        console.error(chalk.red('Error in main function:'), error);
        process.exit(1);
    }
}


// Run the main function
main().catch(error => {
    console.error(chalk.red('Error in main function:'), error);
    process.exit(1);
});