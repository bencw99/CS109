
public class Main 
{
	public static void main(String args[])
	{
		Dataset training = new Dataset("data/vote-train.txt");
		Dataset testing = new Dataset("data/vote-test.txt");
		NaiveBayesClassifier bayesClassifier = new NaiveBayesClassifier(training);
		bayesClassifier.runTests(testing);
		LogisticRegressionClassifier logisticClassifier = new LogisticRegressionClassifier(training, 10000, 0.0001);
		logisticClassifier.runTests(testing);
	}
}
