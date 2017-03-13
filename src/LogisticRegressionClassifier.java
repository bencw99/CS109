
public class LogisticRegressionClassifier extends Classifier
{	
	private int epochs;
	private double learningRate;
	
	private double parameters[];
	
	public LogisticRegressionClassifier(Dataset training, int epochs, double learningRate)
	{
		super(training);
		this.epochs = epochs;
		this.learningRate = learningRate;
		determineParameters();
	}
	
	public void determineParameters()
	{
		this.parameters = new double[this.getTraining().getDimension() + 1];
		for (int i = 0; i < epochs; i ++)
		{
			double gradient[] = new double[this.getTraining().getDimension() + 1];
			for (int dataPoint[] : this.getTraining().getData())
			{
				// Computes the value of z for this data point using current parameters
				double z = 0;
				for (int j = 0; j < parameters.length; j ++)
				{
					z += parameters[j] * (j == 0 ? 1 : dataPoint[j]);
				}
				for (int j = 0; j < gradient.length; j ++)
				{
					gradient[j] += (j == 0 ? 1 : dataPoint[j]) * (dataPoint[0] - 1 / (1 + Math.exp(-z)));
				}
			}
			for (int j = 0; j < gradient.length; j ++)
			{
				parameters[j] += learningRate * gradient[j];
			}
		}
	}
	
	@Override
	public void runTests(Dataset testing)
	{
		int total[] = new int[2];
		int correct[] = new int[2];
		for (int testingData[] : testing.getData())
		{
			total[testingData[0]] ++;
			if (LogisticRegressionClassifier.calculateProbability(this.parameters, testingData, testingData[0]) >= 0.5)
			{
				correct[testingData[0]] ++;
			}
		}
		Classifier.printReport("Logistic Regression Classifier", total, correct);
	}
	
	public static double calculateProbability(double parameters[], int dataPoint[], int result)
	{
		double z = 0;
		for (int i = 0; i < parameters.length; i ++)
		{
			z += parameters[i] * (i == 0 ? 1 : dataPoint[i]);
		}
		return (1 - result) + (2 * result - 1) / (1 + Math.exp(-z));
	}
}
