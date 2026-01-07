package com.dscn;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class AutoGenImage {

    public static final String CHARSET_TRAIN = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ";

    public static void main(String[] args) throws IOException, FontFormatException {

        String[] fonts = new String[]{
                "Arial",
                "Times New Roman",
                "Helvetica",
                "Verdana",
                "Georgia",
                "Garamond",
                "Courier New",
                "Tahoma",
                "Trebuchet MS",
                "Fonts/ARHei.ttf",
                "Fonts/ARIALN.TTF",
                "Fonts/ARIALNB.TTF",
                "Fonts/ARKai_C.TTF",
                "Fonts/ARKai_T.TTF",
                "Fonts/bHEI00M.TTF",
                "Fonts/bHEI01B.TTF",
                "Fonts/bKAI00M.TTF",
                "Fonts/bLEI00D.TTF"
        };

        File directory = new File("D:\\dataset\\wow_ability_icons");
        File[] files = getAllFiles(directory, ".png");

        if (files == null) {
            System.out.println("No files found.");
            return;
        }
        for (File file : files) {
            BufferedImage originalImage = ImageIO.read(file);
            BufferedImage image = resizeImage(originalImage, 50, 50);
            // for char in CHARSET_TRAIN:
            for (int k = 0; k < CHARSET_TRAIN.length(); k++) {
                String text = String.valueOf(CHARSET_TRAIN.charAt(k));
                int fontIdx = new Random().nextInt(fonts.length);
                String font = fonts[fontIdx];
                int fontSize = 12 + new Random().nextInt(3);
                if (font.endsWith(".ttf")) {
                    Font ttf = Font.createFont(Font.TRUETYPE_FONT, new File(font));
                    ttf = ttf.deriveFont((float) fontSize);
                    compositeImage(file, image, ttf, "FONT" + fontIdx, text);
                } else {
                    Font font1 = new Font(font, Font.PLAIN, fontSize);
                    compositeImage(file, image, font1, "FONT" + fontIdx, text);
                }
            }
        }
    }

    private static void compositeImage(File file, BufferedImage image, Font font, String fontType, String text) throws IOException {
        Random random = new Random();
        int i = random.nextInt(26); // 生成0到26之间的随机数
        int j = random.nextInt(26); // 生成0到26之间的随机数
        BufferedImage newImage = image.getSubimage(image.getWidth() - 24 - i, j, 24, 24);
        newImage = copyImage(newImage);
        Graphics2D g2d = newImage.createGraphics();
        // 设置文本颜色
        g2d.setFont(font);
        // 设置字体为不透明
        g2d.setComposite(AlphaComposite.SrcOver);

        Dimension dimension = getFontTextSize(text, font, Color.BLACK);

        // 设置文本位置
        int x = newImage.getWidth() - dimension.width - 3;
        int y = dimension.height + 3;

        drawOutline(g2d, text, x, y);

        // 设置文本颜色为白色并绘制文本
        g2d.setColor(Color.WHITE);
        g2d.drawString(text, x, y);

        newImage = resizeImage(newImage, 128, 32);
        // 保存图片
        String output_path = "traindata/" + file.getName() + "_" + fontType + "_" + (i + 1) * (j + 1) + "_" + text + ".png";
        File outputfile = new File(output_path);
        ImageIO.write(newImage, "png", outputfile);

        // 写入图像路径和对应的字符到gt.txt
        FileWriter fileWriter = new FileWriter("gt.txt", true); // true表示追加模式
        fileWriter.write(output_path + "\t" + text + "\n");
        fileWriter.close();

        g2d.dispose();
    }

    public static BufferedImage resizeImage(BufferedImage originalImage, int width, int height) {
        BufferedImage resizedImage = new BufferedImage(width, height, originalImage.getType());
        Graphics2D g2d = resizedImage.createGraphics();
        g2d.drawImage(originalImage.getScaledInstance(width, height, Image.SCALE_SMOOTH), 0, 0, null);
        g2d.dispose();
        return resizedImage;
    }

    public static BufferedImage copyImage(BufferedImage originalImage) {
        BufferedImage copiedImage = new BufferedImage(originalImage.getWidth(), originalImage.getHeight(), originalImage.getType());
        Graphics2D g2d = copiedImage.createGraphics();
        g2d.drawImage(originalImage, 0, 0, null);
        g2d.dispose();
        return copiedImage;
    }

    public static void drawOutline(Graphics2D g2d, String text, int x, int y) {
        //// 设置文本颜色为黑色并绘制边框
        g2d.setColor(Color.BLACK);
        g2d.drawString(text, x - 1, y - 1);
        g2d.drawString(text, x - 1, y);
        g2d.drawString(text, x - 1, y + 1);
        g2d.drawString(text, x, y - 1);
        g2d.drawString(text, x, y);
        g2d.drawString(text, x, y + 1);
        g2d.drawString(text, x + 1, y - 1);
        g2d.drawString(text, x + 1, y);
        g2d.drawString(text, x + 1, y + 1);
    }

    public static Dimension getFontTextSize(String text, Font font, Color text_color) {
        // 创建一个图形环境
        GraphicsEnvironment ge = GraphicsEnvironment.getLocalGraphicsEnvironment();
        GraphicsDevice gd = ge.getDefaultScreenDevice();
        GraphicsConfiguration gc = gd.getDefaultConfiguration();
        BufferedImage img = gc.createCompatibleImage(32, 32, Transparency.TRANSLUCENT);
        Graphics2D g2d = img.createGraphics();

        // 设置字体和颜色
        g2d.setFont(font);
        g2d.setColor(text_color);

        // 获取字体度量信息
        FontMetrics fm = g2d.getFontMetrics();

        // 测量文本
        int width = fm.stringWidth(text);
        int height = 10;

        // 清理资源
        g2d.dispose();

        // 返回文本尺寸
        return new Dimension(width, height);
    }

    private static File[] getAllFiles(File directory, String extension) {
        java.util.List<File> fileList = new ArrayList<>();
        File[] files = directory.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    fileList.addAll(Arrays.asList(getAllFiles(file, extension)));
                } else if (file.isFile() && file.getName().endsWith(extension)) {
                    fileList.add(file);
                }
            }
        }
        return fileList.toArray(new File[0]);
    }
}