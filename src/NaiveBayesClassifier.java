/**
 * An implementation of the Naive Bayes Classifier in the special case where all variables are binary
 * 
 * @author Benjamin Cohen-Wang
 */
public class NaiveBayesClassifier extends Classifier
{	
	/** The tables storing the counts for each input variable **/
	private int countTables[][][];
	
	/** The probabilities for each input variable using maximum likelihood estimators **/
	private double mleTables[][][];
	
	/** The probabilities for each input variable using laplace estimators **/
	private double leTables[][][];
	
	/**
	 * Invokes superconstructor to initialize training dataset then generates count and probability tables
	 * 
	 * @param training the training dataset to be set
	 */
	public NaiveBayesClassifier(Dataset training)
	{
		super(training);
		this.generateCountTables();
		this.generateProbabilityTables();
	}
	
	/**
	 * Generates the count tables from the training data
	 */
	private void generateCountTables()
	{
		this.countTables = new int[this.getTraining().getDimension()][2][2];
		
		for (int dataPoint[] : this.getTraining().getData())
		{
			for (int i = 1; i < dataPoint.length; i ++)
			{
				this.countTables[i - 1][dataPoint[i]][dataPoint[0]] ++;
			}
		}
	}
	
	/**
	 * Generates the two probability tables from generated count tables
	 */
	private void generateProbabilityTables()
	{
		this.mleTables = new double[this.getTraining().getDimension()][2][2];
		this.leTables = new double[this.getTraining().getDimension()][2][2];

		for (int count = 0; count < this.countTables.length; count ++)
		{
			for (int i = 0; i < this.mleTables[count].length; i ++)
			{
				for (int j = 0; j < this.mleTables[count][0].length; j ++)
				{
					this.mleTables[count][i][j] = (double) this.countTables[count][i][j] / (double) this.getTraining().getSize();
					this.leTables[count][i][j] = (double) (this.countTables[count][i][j] + 1) 
							/ (double) (this.getTraining().getSize() + this.leTables[0].length * this.leTables[0][0].length);
				}
			}
		}
	}
	
	@Override
	public void runTests(Dataset testing)
	{
		int total[] = new int[2];
		int mleCorrect[] = new int[2];
		int leCorrect[] = new int[2];
		for (int testingData[] : testing.getData())
		{
			total[testingData[0]] ++;
			if (NaiveBayesClassifier.calculateProbability(this.mleTables, testingData, testingData[0]) 
					> NaiveBayesClassifier.calculateProbability(this.mleTables, testingData, 1-testingData[0]))
			{
				mleCorrect[testingData[0]] ++;
			}
			if (NaiveBayesClassifier.calculateProbability(this.leTables, testingData, testingData[0]) 
					> NaiveBayesClassifier.calculateProbability(this.leTables, testingData, 1-testingData[0]))
			{
				leCorrect[testingData[0]] ++;
			}
		}
		Classifier.printReport("Maximum Likelihood Estimator", total, mleCorrect);
		Classifier.printReport("Laplace Estimator", total, leCorrect);
	}
	
	/**
	 * Static function to evaluate the probability of the inputs given that the result was observed
	 * 
	 * @param probabilityTables the probability tables from which to draw probabilities of inputs given result
	 * @param inputs the inputs whose probability is being evaluated
	 * @param result the result to be considered as a given when evaluating probabilities
	 * @return the evaluated probability
	 */
	public static double calculateProbability(double probabilityTables[][][], int inputs[], int result)
	{
		double probability = 1.0;
		for (int i = 1; i < inputs.length; i ++)
		{
			// P(X=inputs | Y=result) = P(X=inputs, Y=result) / P(Y=result)
			probability *= probabilityTables[i - 1][inputs[i]][result] / 
					(probabilityTables[i - 1][0][result] + probabilityTables[i - 1][1][result]);
		}
		// P(Y=result)
		probability *= (probabilityTables[0][0][result] + probabilityTables[0][1][result]);
		return probability;
	}
	
	public static void printTables(String name, double tables[][][])
	{
		for (int count = 0; count < tables.length; count ++)
		{
			System.out.println("--------- " + name + " Table " + count + " ---------");
			for (int input = 0; input < tables[count].length; input ++)
			{
				for (int output = 0; output < tables[count][input].length; output ++)
				{
					System.out.print("\t" + tables[count][input][output]);
				}
				System.out.println();
			}
		}
	}
}
