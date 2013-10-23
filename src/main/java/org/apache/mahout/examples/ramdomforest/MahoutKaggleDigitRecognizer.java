package org.apache.mahout.examples.ramdomforest;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Random;

import org.apache.mahout.classifier.df.DecisionForest;
import org.apache.mahout.classifier.df.builder.DefaultTreeBuilder;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.DataLoader;
import org.apache.mahout.classifier.df.data.Instance;
import org.apache.mahout.classifier.df.ref.SequentialBuilder;
import org.apache.mahout.common.RandomUtils;
import org.uncommons.maths.Maths;

public class MahoutKaggleDigitRecognizer {
	
	public static void main(String[] args) throws Exception {
		
		String[] trainDataValues = fileAsStringArray("C:/Users/surenr/git/mahout-ramdom-forest/src/main/java/org/apache/mahout/examples/ramdomforest/train.csv");

		String[] part1 = new String[trainDataValues.length / 10 * 9];
        String[] part2 = new String[trainDataValues.length / 10];

        System.arraycopy(trainDataValues, 0, part1, 0, part1.length);
        System.arraycopy(trainDataValues, part1.length, part2, 0, part2.length);

        trainDataValues = part1;
        String[] testDataValues= part2;
        
		String descriptor = buildDescriptor(trainDataValues[0].split(",").length - 1);
		
		Data data = DataLoader.loadData(DataLoader.generateDataset(descriptor, false, trainDataValues), trainDataValues);

		int numberOfTrees = 100;
		DecisionForest forest = buildForest(numberOfTrees, data);
        
		Data test = DataLoader.loadData(DataLoader.generateDataset(descriptor, false, testDataValues), testDataValues);
		Random rng = RandomUtils.getRandom();

		for (int i = 0; i < test.size(); i++) {
			Instance oneSample = test.get(i);
			
			double actualIndex = oneSample.get(0);
			int actualLabel = data.getDataset().valueOf(0, String.valueOf((int) actualIndex));

			double classify = forest.classify(test.getDataset(), rng, oneSample);
			int label = data.getDataset().valueOf(0, String.valueOf((int) classify));

			System.out.println("label = " + label + " actual = " + actualLabel);
		}
	}

	private static DecisionForest buildForest(int numberOfTrees, Data data) {
		int m = (int) Math.floor(Maths.log(2, data.getDataset().nbAttributes()) + 1);

		DefaultTreeBuilder treeBuilder = new DefaultTreeBuilder();
		treeBuilder.setM(m);

		return new SequentialBuilder(RandomUtils.getRandom(), treeBuilder, data.clone()).build(numberOfTrees);
	}

	private static String[] fileAsStringArray(String file) throws Exception {
		ArrayList<String> list = new ArrayList<String>();

		DataInputStream in = new DataInputStream(new FileInputStream(file));
		BufferedReader br = new BufferedReader(new InputStreamReader(in));

		String strLine;
		br.readLine(); // discard header
		while ((strLine = br.readLine()) != null) {
			list.add(strLine);
		}

		in.close();
		return list.toArray(new String[list.size()]);
	}

	private static String[] testFileAsStringArray(String file) throws Exception {
		ArrayList<String> list = new ArrayList<String>();

		DataInputStream in = new DataInputStream(new FileInputStream(file));
		BufferedReader br = new BufferedReader(new InputStreamReader(in));

		String strLine;
		br.readLine(); // discard top one (header)
		while ((strLine = br.readLine()) != null) {
			list.add("-," + strLine);
		}

		in.close();
		return list.toArray(new String[list.size()]);
	}
	
	private static String buildDescriptor(int numberOfFeatures) {
        StringBuilder builder = new StringBuilder("L ");
        for (int i = 0; i < numberOfFeatures; i++) {
            builder.append("N ");
        }
        return builder.toString();
    }

}