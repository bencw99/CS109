import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;

/**
 * A dataset to be used for training or testing a machine learning algorithm
 * 
 * @author Benjamin Cohen-Wang
 */
public class Dataset 
{
	/* Array of boolean arrays, each representing one data sample where the first element is the dependent variable*/
	private int data[][];

	/**
	 * Constructs this dataset from the data in the given file
	 * @param file the file from which data is to be obtained
	 */
	public Dataset(String file_name)
	{
		File file = new File(file_name);
		
	    try 
	    {
	        Scanner sc = new Scanner(file);
	        
	        int data_dimension = sc.nextInt();
	        int data_size = sc.nextInt();
	        
	        this.data = new int[data_size][data_dimension + 1];
	        
	        for (int line = 0; line < data_size; line ++)
	        {
	        	for (int index = 0; index < data_dimension; index ++)
	        	{
	        		String token = sc.next();
	        		this.data[line][index + 1] = token.charAt(0) - 48;
	        	}
	        	this.data[line][0] = sc.nextInt();
	        }
	        
	        sc.close();
	    } 
	    catch (FileNotFoundException e) 
	    {
	        e.printStackTrace();
	    }
	}
	
	/**
	 * @return the data of this dataset
	 */
	public int[][] getData()
	{
		return this.data;
	}
	
	/**
	 * @return the number of data points in this dataset
	 */
	public int getSize()
	{
		return this.data.length;
	}
	
	/**
	 * @return the number of variables in each data point
	 */
	public int getDimension()
	{
		// One is subtracted to account for the fact that the first element is the dependent variable
		return this.data[0].length - 1;
	}
}
