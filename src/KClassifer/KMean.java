package KClassifer;

/* @author Alpha,Maya,Tharakesh

*/

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class KMean implements IClassifier{

	private int k;
	private ArrayList<Instance<Double>> centroids;
	private ArrayList<String> classifications;
	private ArrayList<String> classes;
	
	public KMean(){
		k = 4;
		centroids = new ArrayList<Instance<Double>>();
		classifications = new ArrayList<String>();
	}
	
	public KMean(int _k){
		k = _k;
		centroids = new ArrayList<Instance<Double>>();
		classifications = new ArrayList<String>();
	}
	
	@Override
	public String classify(Instance<Integer> i) {
		double minDistance = Double.MAX_VALUE;
		int assignedCentroid = -1;
		for (int cIndex = 0; cIndex < centroids.size(); cIndex++) {
			double dist = distance(centroids.get(cIndex), i);
			if (dist < minDistance) {
				minDistance = dist;
				assignedCentroid = cIndex;
			}
		}
			
		return classifications.get(assignedCentroid);
	}
	
	public void learn(Trainingset<Integer> t){
		this.classes = t.getClasses();
		
		initializeCentroids(t);
		Integer[] oldClassification = new Integer[t.size()];
		Integer[] newClassification = new Integer[t.size()];
		do {			
			oldClassification = newClassification.clone();
			newClassification = new Integer[t.size()];
			
			for (int iIndex = 0; iIndex < t.size(); iIndex++) {
				Instance<Integer> instance = t.getInstance(iIndex);

			
				double minDistance = Double.MAX_VALUE;
				int minCentroidIndex = -1;
				for (int cIndex = 0; cIndex < centroids.size(); cIndex++) {
					double dist = distance(centroids.get(cIndex), instance);
					if (dist < minDistance) {
						minDistance = dist;
						minCentroidIndex = cIndex;
					}
				}
				newClassification[iIndex] = minCentroidIndex;
			}
			
			centroids = new ArrayList<Instance<Double>>();
			
			for (int cIndex = 0; cIndex < k; cIndex++) {
				ArrayList<Integer> indexes = new ArrayList<Integer>();
				for (int iIndex = 0; iIndex < t.size(); iIndex++){
					if (newClassification[iIndex] == cIndex)
						indexes.add(iIndex);
				}
				
				ArrayList<Double> features = new ArrayList<Double>();				
				for (int fIndex = 0; fIndex < t.getFeatureCount(); fIndex++){
					int featureSum = 0;
					for (int iocIndex : indexes)
						featureSum += t.getInstance(iocIndex).getFeature(fIndex);
					features.add(((double) featureSum / indexes.size()));
				}
				centroids.add(new Instance<Double>(features, "-"));
			}
			
			
		} while (!Arrays.equals(oldClassification, newClassification));
		
		for (int cIndex = 0; cIndex < k; cIndex++) {
			ArrayList<Integer> indexes = new ArrayList<Integer>();
			for (int iIndex = 0; iIndex < t.size(); iIndex++){
				if (newClassification[iIndex] == cIndex)
					indexes.add(iIndex);
			}
			
			int[] classCount = new int[k];
			for (int iocIndex : indexes) {
				classCount[classes.indexOf(t.getInstance(iocIndex).getClassification())]++;
			}
			List<Integer> classCountArray = Arrays.asList(toObject(classCount));
			classifications.add(classes.get(classCountArray.indexOf(Collections.max(classCountArray))));
		}
	}

	private void initializeCentroids(Trainingset<Integer> t){
		for (int i = 0; i < k; i++) {
			ArrayList<Double> features = new ArrayList<Double>();
			Random rand = new Random();
			for (int fIndex = 0; fIndex < t.getFeatureCount(); fIndex++){
				
				List<Integer> list = Arrays.asList(t.getDomain(fIndex).values);
				features.add((double) rand.nextInt((Collections.max(list) - Collections.min(list) + 1) + Collections.min(list)));
			}
			centroids.add(new Instance<Double>(features, "-"));
		}
	}
	
	private double distance(Instance<Double> centroid, Instance<Integer> i){
		int distance = 0;
		for (int m = 0; m < centroid.getDimension(); m++) {
			distance += Math.abs(centroid.getFeature(m) - i.getFeature(m));
		}
		return distance;
	}
	
	private static Integer[] toObject(int[] intArray) {
		 
		Integer[] result = new Integer[intArray.length];
		for (int i = 0; i < intArray.length; i++) {
			result[i] = Integer.valueOf(intArray[i]);
		}
		return result;
	}
}

