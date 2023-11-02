package org.mif;

import weka.core.Instances;
import java.io.*;
import java.util.ArrayList;

/**
 * renames all the labels of nominal attributes to numbers, they way they
 * appear, e.g., attribute a1 has the labels "what", "so" and "ever" are
 * renamed to "0", "1" and "2".
 *
 * @author FracPete
 */
public class Rename {
  public static void renameLabels(String input, String output, int classIndex) throws Exception {
    // read arff file
    Instances arff = new Instances(new BufferedReader(new FileReader(input)));
    arff.setClassIndex(classIndex);

    var indices = new ArrayList<Integer>();
    for (var i = 0; i < arff.numAttributes(); i++) {
      if (arff.attribute(i).isNominal() && i != classIndex) {
        indices.add(i);
      }
    }

    for (var i: indices) {
      var attribute = arff.attribute(i);

      for (int n = 0; n < attribute.numValues(); n++) {
        arff.renameAttributeValue(attribute, attribute.value(n), "" + n);
      }
    }

    // save arff file
    BufferedWriter writer = new BufferedWriter(new FileWriter(output));
    writer.write(arff.toString());
    writer.newLine();
    writer.flush();
    writer.close();
  }
}
