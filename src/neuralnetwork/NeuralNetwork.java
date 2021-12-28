/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnetwork;

// import com.sun.xml.internal.bind.v2.runtime.unmarshaller.XsiNilLoader;
import java.io.File;  // Import the File class
import java.io.FileNotFoundException;  // Import this class to handle errors
import java.util.Scanner; // Import the Scanner class to read text files
import java.util.ArrayList;

import java.util.Arrays;
import java.util.Collections;

import java.util.Random;

/**
 *
 * @author Magooda
 */
public class NeuralNetwork {

    /**
     * @param args the command line arguments
     */
    static ArrayList<ArrayList> Xs = new ArrayList<>();
    static ArrayList<ArrayList> Ys = new ArrayList<>();

    static int numberofinputs, numberofhidden, numberofoutputs, numberofTrainingExamples, numberofweights;

    static ArrayList<ArrayList> weights = new ArrayList<>();

    static public class record {

        ArrayList<Double> X = new ArrayList<>();
        ArrayList<Double> Y = new ArrayList<>();

        public record(ArrayList x, ArrayList y) {
            X = x;
            Y = y;
        }

    }

    public static void read_from_file(String filename) {
        try {
            File myObj = new File(filename);
            Scanner myReader = new Scanner(myObj);
            numberofinputs = myReader.nextInt();
            numberofhidden = myReader.nextInt();
            numberofoutputs = myReader.nextInt();

            numberofTrainingExamples = myReader.nextInt();
            for (int i = 0; i < numberofTrainingExamples; i++) {
                ArrayList<Double> X = new ArrayList<>();
                ArrayList<Double> Y = new ArrayList<>();
                for (int j = 0; j < numberofinputs; j++) {
                    X.add(myReader.nextDouble());
                }
                for (int j = 0; j < numberofoutputs; j++) {
                    Y.add(myReader.nextDouble());
                }
                Xs.add(X);
                Ys.add(Y);
            }
            myReader.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }

    }

    public static void Normalization() {
        ArrayList<ArrayList> newxs = new ArrayList<>();
        ArrayList<Double> means = new ArrayList<>(Arrays.asList(new Double[numberofinputs]));
        ArrayList<Double> Sd = new ArrayList<>(Arrays.asList(new Double[numberofinputs]));
        Collections.fill(means, 0.0);
        Collections.fill(Sd, 0.0);
        Double value = 0.0, sumTillNow = 0.0, sdTillNow = 0.0;
        for (int i = 0; i < numberofTrainingExamples; i++) {
            for (int j = 0; j < numberofinputs; j++) {
                value = (Double) Xs.get(i).get(j);
                sumTillNow = means.get(j);
                means.remove(j);
                if (i == numberofTrainingExamples - 1) {
                    means.add(j, ((value + sumTillNow) / numberofTrainingExamples));
                } else {
                    means.add(j, (value + sumTillNow));
                }

            }

        }
        for (int i = 0; i < numberofTrainingExamples; i++) {
            for (int j = 0; j < numberofinputs; j++) {

                Double mean = means.get(j);
                value = (Double) Xs.get(i).get(j);
                sdTillNow = Sd.get(j);

                Sd.remove(j);
                if (i == numberofTrainingExamples - 1) {
                    Double Val = (value - mean) * (value - mean) + sdTillNow;
                    Val = Val / numberofTrainingExamples;
                    Sd.add(j, Math.sqrt(Val));
                } else {

                    Sd.add(j, (Double) ((value - mean) * (value - mean)) + sdTillNow);
                }

            }

        }
        for (int i = 0; i < numberofTrainingExamples; i++) {
            ArrayList<Double> newx = new ArrayList<>();
            for (int j = 0; j < numberofinputs; j++) {
                Double new_x = (((Double) Xs.get(i).get(j) - means.get(j)) / Sd.get(j));
                newx.add(new_x);

            }
            newxs.add(newx);
        }

        Xs = newxs;
        System.out.println("Xs"+Xs);

    }

    public static void weights_initialization() {
        numberofweights = numberofinputs * numberofhidden + numberofhidden * numberofoutputs;
        ArrayList<Double> first_layer = new ArrayList<>();
        ArrayList<Double> second_layer = new ArrayList<>();
        Double range = (double) (1.0 / numberofweights);
        for (int j = 0; j < numberofinputs * numberofhidden; j++) {
            first_layer.add(Math.random() * (range + range) - range);
        }
        for (int j = 0; j < numberofhidden * numberofoutputs; j++) {
            second_layer.add(Math.random() * (range + range) - range);
        }
        weights.add(first_layer);
        weights.add(second_layer);
        System.out.println(weights);

    }

    public static ArrayList<ArrayList> FeedForward(int numOfCurrentTrainingExample) {
        ArrayList<Double> Outputs_Out = new ArrayList<>();
        ArrayList<Double> Hidden_Out = new ArrayList<>();
        ArrayList<ArrayList> AllOfTheAbove = new ArrayList<>();   //just for return 
        for (int i = 0; i < numberofhidden; i++) {
            Double H_i_in = 0.0;
            for (int j = 0; j < numberofinputs; j++) {
                H_i_in += (Double) (weights.get(0).get(j)) * ((Double) Xs.get(numOfCurrentTrainingExample).get(j));
            }
            Double Exp = Math.exp(-H_i_in);
            Double H_i_out = (1 / (1 + Exp));
            Hidden_Out.add(H_i_out);
        }

        for (int i = 0; i < numberofoutputs; i++) {
            Double Y_i_in = 0.0;
            for (int j = 0; j < numberofhidden; j++) {
                Y_i_in += (Double) (weights.get(1).get(j)) * ((Double) Ys.get(numOfCurrentTrainingExample).get(j));
            }
            Double Exp = Math.exp(-Y_i_in);
            Double Y_i_out = (1 / (1 + Exp));
            Outputs_Out.add(Y_i_out);
        }
        AllOfTheAbove.add(Hidden_Out);
        AllOfTheAbove.add(Outputs_Out);

        return AllOfTheAbove;

    }

    public static void main(String[] args) {

        ArrayList<ArrayList> Outputs_AllEx = new ArrayList<>();
        ArrayList<ArrayList> Hiddens_AllEx = new ArrayList<>();
        read_from_file("input_3.txt");
        Normalization();
        weights_initialization();

        for (int j = 0; j < numberofTrainingExamples; j++) {
            ArrayList<ArrayList> Outputs_OneEx = new ArrayList<>();
            ArrayList<ArrayList> Hiddens_OneEx = new ArrayList<>();
            ArrayList<ArrayList> Both_Hidden_Out = new ArrayList<>();
            Both_Hidden_Out = FeedForward(j);
            Outputs_OneEx.add(Both_Hidden_Out.get(1));
            Outputs_AllEx.add(Both_Hidden_Out.get(1));

            Hiddens_OneEx.add(Both_Hidden_Out.get(0));
            Hiddens_AllEx.add(Both_Hidden_Out.get(0));

            System.out.println("Training Example number " + (j + 1));
            System.out.println("Hidden outs " + Hiddens_OneEx);
            System.out.println("Y outs " + Outputs_OneEx);
            System.out.println("###############################");

        }

    }
}
