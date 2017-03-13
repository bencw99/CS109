/**
 * A superclass for various classification machine learning algorithms
 * 
 * @author Benjamin Cohen-Wang
 */
public abstract class Classifier 
{
	/** The training dataset used for this classifier **/
	private Dataset training;
	
	/**
	 * A simple constructor to initialize the training and testing datasets
	 * 
	 * @param training the training dataset to be set
	 */
	public Classifier(Dataset training)
	{
		this.training = training;
	}
	
	/**
	 * @return the training dataset of this classifier
	 */
	public Dataset getTraining()
	{
		return this.training;
	}
	
	/**
	 * Computes predictions for given testing dataset, compares with actual results, and reports correctness/accuracy
	 * 
	 * @param testing the testing dataset to be used
	 */
	public abstract void runTests(Dataset testing);
	
	/**
	 * A static function to print a report with the given statistics
	 * 
	 * @param name the name of the report being printed (the classifier used)
	 * @param total the total number of data points for each class
	 * @param correct the number of correctly identified data points for each class
	 */
	public static void printReport(String name, int total[], int correct[])
	{
		int titleSize = 20;
		for (int i = 0; i < titleSize - name.length() / 2; i ++) System.out.print("-");
		System.out.print(" " + name + " ");
		for (int i = 0; i < titleSize - (name.length() - 1) / 2; i ++) System.out.print("-");
		System.out.println();
		for (int i = 0; i < total.length; i ++)
		{
			System.out.println("Class " + i + ": tested " + total[i] + ", correctly classified " + correct[i]);
		}
		System.out.println("Overall: tested " + (total[0] + total[1]) + ", correctly classified " + (correct[0] + correct[1]));
		System.out.println("Accuracy: " + ((double) (correct[0] + correct[1])) / ((double) (total[0] + total[1])));
		System.out.println();
	}
}
